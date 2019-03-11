from dataloaders.datasets.pascal import VOCSegmentation
from modeling.correction_net.sbox_net import SBoxNet
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


class ClickDataset(VOCSegmentation):
    def __init__(self, sbox_net=None, split='train', transform=None):
        if not sbox_net:
            sbox_net = utils.load_model(SBoxNet(), '../../run/model_best.pth.tar')

        if not transform:
            transform = transforms.Compose([
                tr.CropFromMask(crop_elems=('image', 'gt'), relax=20, zero_pad=True),
                tr.FixedResize(resolutions={'crop_image': (256, 256), 'crop_gt': (256, 256)}),
                tr.Normalize(elems='crop_image'),
            ])
        VOCSegmentation.__init__(self,
                                 root=Path.db_root_dir('pascal'),
                                 split=split,
                                 transform=transform,
                                 download=False,
                                 preprocess=False,
                                 area_thres=500,
                                 retname=True,
                                 suppress_void_pixels=True,
                                 default=False)
        self.sbox_net = sbox_net
        self.filtered_obj_file = os.path.join(self.root, self.BASE_DIR, 'ImageSets', 'Segmentation',
                                              '_'.join(self.split) + '_instances_filtered_' + self.area_th_str + '.txt')
        if not self._check_filtered_file():
            self._filter_obj_list()

        # change the obj list of base class, in hope it will consequently change the behaviour of 'get_item' method
        # of base class so that we dont need to reinvent the wheel
        print('obj list len before:{}'.format(len(self.obj_list)))
        self.obj_list = self.filtered_obj_list
        print('obj list len after filter:{}'.format(len(self.obj_list)))

    def _check_filtered_file(self):
        if not os.path.isfile(self.filtered_obj_file):
            return False
        else:
            self.filtered_obj_list = json.load(open(self.filtered_obj_file, 'r'))
            return True

    def _filter_obj_list(self, th=500):
        self.filtered_obj_list = []
        for i in range(len(self.obj_list)):
            sample = super().__getitem__(i)
            inputs = torch.from_numpy(sample['crop_image'].transpose((2, 0, 1))[np.newaxis, ...])
            pred, pfm = self.sbox_net(inputs)
            pred = pred.data.cpu().numpy()

            pred = np.argmax(pred, axis=1)[0].astype(np.uint8)
            gt = sample['crop_gt']
            FPs = pred > gt
            FNs = pred < gt
            if FPs.sum() >= th or FNs.sum() >= th:
                self.filtered_obj_list.append(self.obj_list[i])

        with open(self.filtered_obj_file, 'w') as outfile:
            outfile.write(json.dumps(self.filtered_obj_list))

        print('total {} objects need to be refined by clicknet'.format(len(self.filtered_obj_list)))

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        inputs = torch.from_numpy(sample['crop_image'].transpose((2, 0, 1))[np.newaxis, ...])
        pred, fpm = self.sbox_net(inputs)
        sample['fpm'] = fpm.data.cpu().numpy()
        pred_origin = pred.data.cpu().numpy()
        pred = np.argmax(pred_origin, axis=1)[0].astype(np.uint8)
        sample['pred'] = pred
        sample['pred_origin'] = pred_origin
        sample = self._simulate_user_interaction(sample)
        return sample

    def _simulate_user_interaction(self, sample):
        def __gen_EDM(mask):
            mask = ndimage.binary_erosion(mask, structure=np.ones((5, 5))).astype(FPs.dtype)
            label, num = ndimage.label(mask)
            edm = np.zeros(mask.shape)
            if num != 0:
                max_label = np.argmax(ndimage.sum(mask, label, range(1, num + 1))) + 1
                idx_xs, idx_ys = np.where(label == max_label)
                _tmp = np.random.randint(0, len(idx_xs))
                shape = mask.shape
                edm = np.zeros(shape, dtype=np.float32)
                edm[...] = 255
                xs = np.arange(0, shape[1], 1, np.float)
                ys = np.arange(0, shape[0], 1, np.float)
                ys = ys[:, np.newaxis]
                edm = np.sqrt((xs - idx_xs[_tmp]) ** 2 + (ys - idx_ys[_tmp]) ** 2)
                edm[edm > 255] = 255
                edm = 255 - edm
            return edm / 255

        pred, gt = sample['pred'], sample['crop_gt']
        FPs = (pred > gt).astype(np.uint8)
        FNs = (pred < gt).astype(np.uint8)
        sample['neg_map'] = __gen_EDM(FPs)
        sample['pos_map'] = __gen_EDM(FNs)
        return sample



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch
    from torchvision import transforms
    import cv2

    dataset = ClickDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    np.random.seed(42)

    for i, sample in enumerate(dataloader):
        while 1:
            cv2.imshow('image', sample['crop_image'].numpy()[0])
            cv2.imshow('crop_gt', sample['crop_gt'].numpy()[0])
            pred = sample['pred'].numpy()[0]
            FNs = (sample['crop_gt'].numpy()[0] > pred).astype(np.uint8)
            FNs = FNs * 255
            FPs = (sample['crop_gt'].numpy()[0] < pred).astype(np.uint8)
            FPs = FPs * 255
            pred = pred * 255
            fp_eroded = sample['neg_map'].numpy()[0]
            # cv2.imshow('pred', pred)
            # cv2.imshow('FPs ', FPs)
            # cv2.imshow('FNs ', FNs)
            cv2.imshow('neg_map ', fp_eroded)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        break

    plt.show(block=True)
