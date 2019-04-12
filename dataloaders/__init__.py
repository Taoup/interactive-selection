from dataloaders.datasets import cityscapes, combine_dbs, pascal, sbd, click_dataset
# from dataloaders.datasets import coco
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloaders import custom_transforms as tr

def make_data_loader(args, **kwargs):
    crop_size = args.crop_size
    if args.dataset == 'pascal':
        composed_transforms_tr = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25)),
            tr.CropFromMask(crop_elems=('image', 'gt'), relax=20, zero_pad=True),
            tr.FixedResize(resolutions={'crop_image': (crop_size, crop_size), 'crop_gt': (crop_size, crop_size)}),
            tr.Normalize(elems='crop_image'),
            tr.ToTensor()
        ])
        composed_transforms_val = transforms.Compose([
            tr.CropFromMask(crop_elems=('image', 'gt'), relax=20, zero_pad=True, jitters_bound=None),
            tr.FixedResize(resolutions={'crop_image': (crop_size, crop_size), 'crop_gt': (crop_size, crop_size)}),
            tr.Normalize(elems='crop_image'),
            tr.ToTensor()])
        train_set = pascal.VOCSegmentation(split='train', transform=composed_transforms_tr)
        val_set = pascal.VOCSegmentation(split='val', transform=composed_transforms_val)
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        NUM_CLASSES = 2
        return train_loader, val_loader, test_loader, NUM_CLASSES

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'coco':
        train_set = coco.COCOSegmentation(split='train')
        val_set = coco.COCOSegmentation(split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'click':
        tr_transform = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25)),
            tr.CropFromMask(crop_elems=('image', 'gt'), jitters_bound=(5, 30)),
            tr.FixedResize(resolutions={'crop_image': (crop_size, crop_size), 'crop_gt': (crop_size, crop_size)}),
            tr.Normalize(elems='crop_image'),
            tr.ToTensor()
        ])
        val_transform = transforms.Compose([
            tr.CropFromMask(crop_elems=('image', 'gt'), relax=20, zero_pad=True, jitters_bound=None),
            tr.FixedResize(resolutions={'crop_image': (crop_size, crop_size), 'crop_gt': (crop_size, crop_size)}),
            tr.Normalize(elems='crop_image'),
            tr.ToTensor(),
        ])
        train_set = pascal.VOCSegmentation(split='train', transform=tr_transform)
        train_set.reset_target_list(args)
        val_set = pascal.VOCSegmentation(split='val', transform=val_transform)
        num_class = 2
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError

