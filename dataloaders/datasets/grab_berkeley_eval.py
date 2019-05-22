import torch, cv2
import os
import numpy as np

import torch.utils.data as data
from PIL import Image
from mypath import Path
import dataloaders.custom_transforms as tr


class GrabBerkely(data.Dataset):
    def __init__(self,
                 which='grabcut',
                 transform=None,
                 suppress_void_pixels=True,
                 default=False):

        self.root = Path.db_root_dir(which)
        _mask_dir = os.path.join(self.root, 'masks')
        _image_dir = os.path.join(self.root, 'images')
        self.transform = transform
        self.suppress_void_pixels = suppress_void_pixels
        self.default = default
        self.obj_list_file = os.path.join(self.root, 'instances.txt')
        self.suffix = '.png'
        if which == 'grabcut':
            self.suffix = '.bmp'

        self.images = []
        self.masks = []

        with open(os.path.join(self.root, 'name.txt'), "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            _image = os.path.join(_image_dir, line + ".jpg")
            _mask = os.path.join(_mask_dir, line + self.suffix)
            assert os.path.isfile(_image)
            assert os.path.isfile(_mask)
            self.images.append(_image)
            self.masks.append(_mask)

        assert (len(self.images) == len(self.masks))

        print('Number of images: {:d}'.format(len(self.masks)))

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'gt': _target}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.masks)

    def _make_img_gt_point_pair(self, index):
        image_path = self.images[index]
        mask_path = self.masks[index]

        # Read Image
        _img = np.array(Image.open(image_path).convert('RGB')).astype(np.float32)

        # Read Target object
        if self.suffix == '.bmp':
            _tmp = (np.array(Image.open(mask_path))).astype(np.float32)
        else:
            _tmp = (np.array(Image.open(mask_path).convert('L'))).astype(np.float32)
        _void_pixels = (_tmp == 128)
        _tmp[_void_pixels] = 255
        _target = (_tmp != 0).astype(np.float32)

        return _img, _target


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch
    from torchvision import transforms

    transform = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25)),
        tr.CropFromMask(crop_elems=('image', 'gt'), relax=30, zero_pad=True, jitters_bound=(50, 51)),
        tr.FixedResize(resolutions={'crop_image': (513, 513), 'crop_gt': (513, 513)}),
        tr.Normalize(elems='crop_image'),
        # tr.Normalize(elems='image'),
    ])
    transform_t = transforms.Compose([
        tr.Normalize(elems='image'),
    ])

    # dataset = GrabBerkely(which='bekeley', transform=transform)
    dataset = GrabBerkely(which='grabcut', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    for i, sample in enumerate(dataloader):
        while 1:
            cv2.imshow('image', sample['crop_image'][0].numpy())
            cv2.imshow('crop_gt', sample['crop_gt'][0].numpy())
            # cv2.imshow('image', sample['image'][0].numpy())
            # cv2.imshow('gt', sample['gt'][0].numpy())
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

        if i == 4:
            break

    plt.show(block=True)
