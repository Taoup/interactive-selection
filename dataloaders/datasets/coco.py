import numpy as np
from torch.utils.data import Dataset
import os
from pycocotools.coco import COCO
from torchvision import transforms
from dataloaders import custom_transforms as tr
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class COCOSegmentation(Dataset):
    NUM_CLASSES = 2

    def __init__(self, split='train'):
        super().__init__()
        base_dir = '/data/mht/datasets/croudAI'
        ann_file = os.path.join(base_dir, '{}/annotation.json'.format(split))
        self.img_dir = os.path.join(base_dir, '{}/images'.format(split))
        self.split = split
        self.coco = COCO(ann_file)
        self.annIds = self.coco.getAnnIds()
        self.bigAnnIds = self._filteredAnns(8000)

    def _filteredAnns(self, thresh=5000):
        anns = self.coco.loadAnns(self.annIds)
        big_area_anns = [ann for ann in anns if ann['area'] > thresh]
        ids = [ann['id'] for ann in big_area_anns]
        return ids

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'gt': _target}

        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        coco = self.coco
        annId = self.bigAnnIds[index]
        anns = coco.loadAnns(annId)
        img_id = anns[0]['image_id']
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        _img = np.array(Image.open(os.path.join(self.img_dir, path)).convert('RGB'))
        _target = coco.annToMask(anns[0])

        return _img, _target

    def transform_tr(self, sample):
        composed_transforms_tr = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25)),
            tr.CropFromMask(crop_elems=('image', 'gt'), relax=20, zero_pad=True),
            tr.FixedResize(resolutions={'crop_image': (300, 300), 'crop_gt': (300, 300)}),
            tr.SimUserInput(no_exp=True),
            tr.Normalize(elems='crop_image'),
            tr.ToTensor()
        ])
        return composed_transforms_tr(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.CropFromMask(crop_elems=('image', 'gt'), relax=20, zero_pad=True),
            tr.FixedResize(resolutions={'crop_image': (300, 300), 'crop_gt': (300, 300)}),
            tr.SimUserInput(no_exp=True),
            tr.Normalize(elems='crop_image'),
            tr.ToTensor()])

        return composed_transforms(sample)


    def __len__(self):
        return len(self.bigAnnIds)



if __name__ == "__main__":
    from dataloaders import custom_transforms as tr
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt

    coco_val = COCOSegmentation(split='train')
    print(len(coco_val))

    dataloader = DataLoader(coco_val, batch_size=4, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["crop_image"].size()[0]):
            img = sample['crop_image'].numpy()[0]
            img = np.transpose(img, axes=[1, 2, 0])
            gt = sample['crop_gt'].numpy()[0]
            neg = sample['neg_map'].numpy()[0][0]
            print(gt.shape)
            print(img.shape)
            plt.title('display')
            plt.subplot(311)
            plt.imshow(img)
            plt.subplot(312)
            plt.imshow(gt)
            plt.subplot(313)
            plt.imshow(neg)

        if ii == 1:
            break

    plt.show(block=True)