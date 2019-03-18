from modeling.correction_net.click_net import *
from modeling.correction_net.sbox_net import *
import torch.nn as nn
import numpy as np
from scipy import ndimage


class FusionNet(nn.Module):
    def __init__(self, sbox=None, click=None, sigma=50):
        super(FusionNet, self).__init__()
        self.sbox_net = sbox
        self.click_net = click
        self.sigma = sigma

    def forward(self, image, cat=False, res=False, **input):
        sbox_pred, pyramid_feat_maps = self.sbox_net(image)
        pos_map, neg_map = self._simulate_user_interaction(sbox_pred, input['crop_gt'])
        if image.is_cuda:
            pos_map, neg_map = pos_map.cuda(), neg_map.cuda()
        simulated_clicks = torch.cat([neg_map, pos_map], dim=1)
        if cat:  # 需要从长计议，对Click-Net有影响
            together = torch.cat([simulated_clicks, sbox_pred])
        else:
            together = simulated_clicks + sbox_pred
        click_pred = self.click_net(together, pyramid_feat_maps)
        if res:
            return click_pred + sbox_pred, click_pred  # explicitly learn to correct mistakes made by sbox-net
        else:
            return click_pred, sbox_pred

    def _simulate_user_interaction(self, pred, gt):
        _pred = torch.argmax(pred, dim=1).int()
        _gt = gt.int()
        self.FPs = (_pred > _gt)
        self.FNs = (_pred < _gt)
        pos_map = self.__gen_EDM(self.FNs)
        neg_map = self.__gen_EDM(self.FPs)
        return pos_map, neg_map

    def __gen_EDM(self, mask, limit=4):
        edm_list = []
        for i in range(mask.shape[0]):
            tmp = mask[i].cpu()
            tmp = ndimage.binary_erosion(tmp, structure=np.ones((3, 3))).astype(np.uint8)
            self.eroded = tmp
            label, num = ndimage.label(tmp)
            shape = tmp.shape
            udm = np.zeros(shape, dtype=np.float32)
            if num != 0:
                sorted_labels = np.argsort(ndimage.sum(tmp, label, range(1, num + 1)))
                for i in range(len(sorted_labels)):
                    if i >= limit:
                        break
                    cur_label = sorted_labels[-(i + 1)] + 1  # Caution for BUG: -(i+1) is crucial
                    if (label == cur_label).sum() > 100:
                        idx_xs, idx_ys = np.where(label == cur_label)
                        _tmp = np.random.randint(0, len(idx_xs))  # TODO: find the center of ill-segmented region
                        xs = np.arange(0, shape[1], 1, np.float)
                        ys = np.arange(0, shape[0], 1, np.float)
                        ys = ys[:, np.newaxis]
                        euclid = np.exp(
                            -4 * np.log(2) * ((xs - idx_ys[_tmp]) ** 2 + (ys - idx_xs[_tmp]) ** 2) / self.sigma ** 2)
                        udm = np.maximum(euclid, udm)
                    else:
                        break
            edm_list.append(torch.from_numpy(udm[np.newaxis, np.newaxis, ...].astype(np.float32)))

        return torch.cat(edm_list, dim=0)


if __name__ == '__main__':
    from torchvision import transforms
    import dataloaders.custom_transforms as tr
    from dataloaders.datasets.pascal import VOCSegmentation
    import cv2

    DEBUG = True
    sbox = SBoxNet()
    sbox.load('../../run/sbox_miou_7686.pth.tar')
    fusion = FusionNet(sbox, ClickNet())
    transform = transforms.Compose([
        tr.CropFromMask(crop_elems=('image', 'gt'), relax=20, zero_pad=True),
        tr.FixedResize(resolutions={'crop_image': (256, 256), 'crop_gt': (256, 256)}),
        tr.Normalize(elems='crop_image'),
        tr.ToTensor()
    ])

    dataset = VOCSegmentation(split=['train', 'val'], transform=transform, retname=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for i, sample in enumerate(dataloader):
        with torch.no_grad():
            pred = fusion(sample['crop_image'], crop_gt=sample['crop_gt'])
        _pred = torch.argmax(pred, dim=1).int()
        _gt = sample['crop_gt'].int()
        print(fusion.FPs.shape, fusion.FPs.dtype)

        pred_img = np.argmax(pred.data.numpy(), axis=1)[0].astype(np.uint8)
        img = sample['crop_image'][0].numpy()
        img = np.transpose(img, [1, 2, 0])
        gt = sample['crop_gt'][0].data.numpy()
        eroded = fusion.eroded
        print(eroded.dtype, eroded.shape)
        print(gt.dtype)
        while 1:
            cv2.imshow('image', img)
            cv2.imshow('pred', pred_img * 255)
            cv2.imshow('crop_gt', gt)
            cv2.imshow('false pos', fusion.FPs[0].numpy() * 255)
            cv2.imshow('eroded', fusion.eroded * 255)
            cv2.imshow('false neg', fusion.FNs[0].numpy() * 255)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        if i == 1:
            break
