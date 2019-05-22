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

    def __init__(self, split='ral', cat='seen'):
        super().__init__()
        base_dir = '/data/mht/datasets/coco'
        ann_file = os.path.join(base_dir, 'annotations', 'instances_{}2017.json'.format(split))
        self.img_dir = os.path.join(base_dir, '{}2017/'.format(split))
        self.split = split
        self.coco = COCO(ann_file)
        self.pascal_cats = ['airplane', 'bicycle', 'bird', 'boat', 'bottle',
                            'bus', 'car', 'cat', 'chair', 'cow',
                            'dining table', 'dog', 'horse', 'motorcycle', 'person',
                            'potted plant', 'sheep', 'couch', 'train', 'tv']

        all_coco_cats = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                         "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                         "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                         "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                         "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                         "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                         "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                         "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                         "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                         "teddy bear", "hair drier", "toothbrush"]
        self.unseen_cats = set(all_coco_cats) - set(self.pascal_cats)
        self.sample_objects()
        if cat == 'seen':
            self.annIds = self.seen_ann_ids
        else:
            self.annIds = self.unseen_ann_ids

    def sample_objects(self, thresh=8000):
        np.random.seed(33)
        seen_cat_ids = self.coco.getCatIds(self.pascal_cats)
        unseen_cat_ids = self.coco.getCatIds(self.unseen_cats)
        self.seen_ann_ids = []
        for id in seen_cat_ids:
            ann_ids = self.coco.getAnnIds(catIds=id)
            anns = self.coco.loadAnns(ann_ids)
            big_area_anns = [ann for ann in anns if ann['area'] > thresh]
            ids = [ann['id'] for ann in big_area_anns]
            indice = np.random.randint(0, len(ids), 10)
            self.seen_ann_ids = self.seen_ann_ids + [ids[i] for i in indice]

        self.unseen_ann_ids = []
        for id in unseen_cat_ids:
            ann_ids = self.coco.getAnnIds(catIds=id)
            anns = self.coco.loadAnns(ann_ids)
            big_area_anns = [ann for ann in anns if ann['area'] > thresh]
            ids = [ann['id'] for ann in big_area_anns]
            indice = np.random.randint(0, len(ids), 4)
            self.unseen_ann_ids = self.unseen_ann_ids + [ids[i] for i in indice]

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'gt': _target}

        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        coco = self.coco
        annId = self.annIds[index]
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
            tr.CropFromMask(crop_elems=('image', 'gt'), relax=20, zero_pad=True, jitters_bound=(50, 51)),
            tr.FixedResize(resolutions={'crop_image': (513, 513), 'crop_gt': (513, 513)}),
            tr.Normalize(elems='crop_image'),
            tr.ToTensor()
        ])
        return composed_transforms_tr(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.CropFromMask(crop_elems=('image', 'gt'), relax=20, zero_pad=True, jitters_bound=(50, 51)),
            tr.FixedResize(resolutions={'crop_image': (513, 513), 'crop_gt': (513, 513)}),
            tr.Normalize(elems='crop_image'),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __len__(self):
        return len(self.annIds)


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
