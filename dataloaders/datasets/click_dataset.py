from dataloaders.datasets.pascal import VOCSegmentation
from mypath import Path
import dataloaders.custom_transforms as tr
import torch
import numpy as np
import json
import os
from torchvision import transforms


class ClickDataset(VOCSegmentation):
    def __init__(self, sbox_net, split='train', transform=None):
        VOCSegmentation.__init__(self,
                                 root=Path.db_root_dir('pascal'),
                                 split=split,
                                 transform=transform,
                                 download=False,
                                 preprocess=False,
                                 area_thres=2500,
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
            pred = self.sbox_net(inputs)
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
        pred = self.sbox_net(inputs)
        pred = pred.data.cpu().numpy()
        sample['pred'] = pred
        return sample


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch
    from torchvision import transforms
    from dataloaders import utils
    from modeling.correction_net.sbox_net import SBoxNet
    import cv2

    sbox_net = utils.load_model(SBoxNet(), '../../run/model_best.pth.tar')

    composed_transforms_tr = transforms.Compose([tr.CropFromMask(crop_elems=('image', 'gt'), relax=20, zero_pad=True),
                                                 tr.FixedResize(
                                                     resolutions={'crop_image': (256, 256), 'crop_gt': (256, 256)}),
                                                 tr.Normalize(elems='crop_image'),
                                                 ])

    dataset = ClickDataset(sbox_net, transform=composed_transforms_tr)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for i, sample in enumerate(dataset):
        while 1:
            cv2.imshow('image', np.array(sample['crop_image']))
            cv2.imshow('crop_gt', sample['crop_gt'])
            pred = sample['pred']
            pred = np.argmax(pred, axis=1)[0].astype(np.uint8)
            FNs = (sample['crop_gt'] > pred).astype(np.uint8)
            FNs = FNs * 255
            FPs = (sample['crop_gt'] < pred).astype(np.uint8)
            FPs = FPs * 255
            pred = pred * 255
            cv2.imshow('pred', pred)
            cv2.imshow('FPs ', FPs)
            cv2.imshow('FNs ', FNs)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        if i == 4:
            break

    plt.show(block=True)
