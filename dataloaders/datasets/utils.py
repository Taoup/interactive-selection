import torch
from mypath import Path
from dataloaders.datasets.pascal import VOCSegmentation
from torchvision import transforms
import dataloaders.custom_transforms as tr
from modeling.deeplab1 import DeepLabX
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
import json
import os


def _generate_matrix(gt_image, pre_image, num_class=2):
    mask = (gt_image >= 0) & (gt_image < num_class)
    label = num_class * gt_image[mask].astype('int') + pre_image[mask]
    count = np.bincount(label, minlength=num_class ** 2)
    confusion_matrix = count.reshape(num_class, num_class)
    return confusion_matrix


def Mean_Intersection_over_Union(confusion_matrix):
    MIoU = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix))
    MIoU = np.nanmean(MIoU)
    return MIoU


def compute_iou(y_pred, y_true):
    # ytrue, ypred is a flatten vector
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    iou = intersection / union.astype(np.float32)
    return np.mean(iou)


def extract_hard_example(args, batch_size=8, recal=False):
    transform = transforms.Compose([
        tr.CropFromMask(crop_elems=('image', 'gt'), relax=20, zero_pad=True, jitters_bound=(40, 70)),
        tr.FixedResize(resolutions={'crop_image': (513, 513), 'crop_gt': (513, 513)}),
        tr.Normalize(elems='crop_image'),
        tr.ToTensor(),
    ])
    dataset = VOCSegmentation(
        root=Path.db_root_dir('pascal'),
        split='train',
        transform=transform,
        download=False,
        preprocess=False,
        area_thres=500,
        retname=True,
        suppress_void_pixels=True,
        which_part=args.which,
        default=False)
    click_list_file = os.path.join(dataset.root, dataset.BASE_DIR, 'ImageSets', 'Segmentation',
                                   '-'.join([args.sbox, 'mIoU_thres', str(args.low_thres), str(args.high_thres), 'on',
                                             args.which, 'sets']) + '.txt')
    if os.path.isfile(click_list_file) and not recal:
        return
    device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
    sbox_net = DeepLabX(pretrain=False)
    path = 'run/' + args.sbox
    sbox_net.load_state_dict(torch.load(path, map_location=device)['state_dict'])
    sbox_net = sbox_net.to(device)
    sbox_net.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    tbar = tqdm(dataloader, desc='\r')
    hard_examples = []
    n_hard = 0
    for i, sample in enumerate(tbar):
        image, gt = sample['crop_image'], sample['crop_gt']
        image, gt = image.to(device), gt.to(device)
        with torch.no_grad():
            pred, _, _ = sbox_net(image)

        pred = F.interpolate(pred, size=gt.size()[-2:], mode='bilinear', align_corners=True)
        pred = pred.data.cpu().numpy()
        target = gt.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        for j in range(pred.shape[0]):
            matrix = _generate_matrix(target[j], pred[j])
            iou = Mean_Intersection_over_Union(matrix)
            if iou <= args.high_thres and iou >= args.low_thres:
                print('{}th object, IoU:{}, needs to be refined, {}th object added'.format(i * batch_size + j, iou,
                                                                                           n_hard))
                n_hard += 1
                hard_examples.append(dataset.target_list[i * batch_size + j])

    with open(click_list_file, 'w') as outfile:
        outfile.write(json.dumps(hard_examples))

    print('total {} objects need to be refined by clicknet'.format(len(hard_examples)))


if __name__ == '__main__':
    class args:
        pass


    args.sbox = 'sbox_miou_8527.pth.tar'
    args.low_thres = .70
    args.high_thres = .95
    args.which = 'whole'
    extract_hard_example(args)
