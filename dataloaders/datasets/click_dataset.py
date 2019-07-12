from dataloaders.datasets.pascal import VOCSegmentation
from modeling.deeplab import DeepLabX
from mypath import Path
import dataloaders.custom_transforms as tr
import torch
import numpy as np
import json
import os
from torchvision import transforms
from PIL import Image, ImageFilter
from scipy import ndimage
from dataloaders import utils
from sklearn.metrics import confusion_matrix


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

class ClickDataset(VOCSegmentation):
    def __init__(self, sbox_net=None, split='train', transform=None,
                 sbox='sbox_513_8925.pth.tar',
                 miou_thres=0.85,
                 which='click'):
        sbox_net = utils.load_model(DeepLabX(pretrain=False), 'run/sbox/' + sbox)

        VOCSegmentation.__init__(self,
                                 root=Path.db_root_dir('pascal'),
                                 split=split,
                                 transform=transform,
                                 download=False,
                                 preprocess=False,
                                 area_thres=500,
                                 retname=True,
                                 suppress_void_pixels=True,
                                 which_part=which,
                                 default=False)
        # sbox_net.eval()
        self.sbox_net = sbox_net
        self.device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
        self.sbox_net = self.sbox_net.to(self.device)
        self.sbox_net.eval()

        self.miou_thres = miou_thres
        self.click_list_file = os.path.join(self.root, self.BASE_DIR, 'ImageSets', 'Segmentation',
                                            '-'.join(
                                                [sbox, 'mIoU_thres', str(miou_thres), 'on', which, 'sets']) + '.txt')
        np.random.seed(42)
        if split == 'train' and not self._check_filtered_file():
            pre_transform = transforms.Compose([
                tr.CropFromMask(crop_elems=('image', 'gt'), relax=20, zero_pad=True, jitters_bound=None),
                tr.FixedResize(resolutions={'crop_image': (512, 512), 'crop_gt': (512, 512)}),
                tr.Normalize(elems='crop_image'),
                tr.ToTensor(),
            ])
            tmp = self.transform
            self.transform = pre_transform
            self._filter_obj_list()
            self.transform = tmp

        np.random.seed(42)
        # change the obj list of base class, in hope it will consequently change the behaviour of 'get_item' method
        # of base class so that we dont need to reinvent the wheel
        print('target list len:{}'.format(len(self.target_list)))

    def _check_filtered_file(self):
        if not os.path.isfile(self.click_list_file):
            return False
        else:
            self.filtered_obj_list = json.load(open(self.click_list_file, 'r'))
            print('obj list len before:{}'.format(len(self.target_list)))
            self.target_list = self.filtered_obj_list
            return True

    def _filter_obj_list(self):
        self.filtered_obj_list = []
        j = 0
        for i in range(len(self.target_list)):
            # for i in range(50):
            sample = super().__getitem__(i)
            inputs = sample['crop_image'].unsqueeze(0)
            inputs = inputs.to(self.device)
            with torch.no_grad():
                pred, _, _ = self.sbox_net(inputs)
            pred = pred.data.cpu().numpy()

            pred = np.argmax(pred, axis=1)[0].astype(np.uint8)
            gt = sample['crop_gt'].data.cpu().numpy()
            iou = compute_iou(pred, gt)
            if iou <= self.miou_thres:
                print('{}th object, IoU:{}, needs to be refined, {}th object added'.format(i, iou, j))
                j += 1
                self.filtered_obj_list.append(self.target_list[i])

        with open(self.click_list_file, 'w') as outfile:
            outfile.write(json.dumps(self.filtered_obj_list))

        print('total {} objects need to be refined by clicknet'.format(len(self.filtered_obj_list)))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch
    from torchvision import transforms
    import torch.nn.functional as F
    import cv2

    dataset = ClickDataset(which='click', transform=transform)
    model = dataset.sbox_net
    batch_size = 4
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for i, sample in enumerate(dataloader):
        images = sample['crop_image'].to(dataset.device)
        with torch.no_grad():
            pred, _, _ = model(images)
        pred = pred.data.cpu().numpy()
        print('OKKKKKK')
        for i in range(batch_size):
            pred_i = np.argmax(pred, axis=1)[i].astype(np.uint8)
            # pred = F.softmax(pred)[0][0].numpy()
            gt_i = sample['crop_gt'].data.cpu().numpy()[i]
            print('miou:{}'.format(compute_iou(pred_i, gt_i)))
            # images_i = np.transpose(images.data.cpu().numpy()[i], [1, 2, 0])
            # while 1:
            #     cv2.imshow('image', images_i)
            #     cv2.imshow('crop_gt', gt_i)
            #     cv2.imshow('pred', pred_i*255)
            #     key = cv2.waitKey(1) & 0xff
            #     if key == ord('q'):
            #         exit(0)
            #     elif key == ord(' '):
            #         break
