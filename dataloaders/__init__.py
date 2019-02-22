from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloaders import custom_transforms as tr

def make_data_loader(args, **kwargs):

    if args.dataset == 'pascal':
        composed_transforms_tr = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25)),
            tr.CropFromMask(crop_elems=('image', 'gt'), relax=20, zero_pad=True),
            tr.FixedResize(resolutions={'crop_image': (512, 512), 'crop_gt': (512, 512)}),
            tr.SimUserInput(no_exp=True),
            tr.Normalize(elems='crop_image'),
            tr.ConcatInputs(elems=('crop_image', 'neg_map', 'pos_map')),
            tr.ToTensor()
        ])
        composed_transforms_val = transforms.Compose([
            tr.CropFromMask(crop_elems=('image', 'gt'), relax=20, zero_pad=True),
            tr.FixedResize(resolutions={'crop_image': (512, 512), 'crop_gt': (512, 512)}),
            tr.SimUserInput(no_exp=True),
            tr.Normalize(elems='crop_image'),
            tr.ConcatInputs(elems=('crop_image', 'neg_map', 'pos_map')),
            tr.ToTensor()])
        train_set = pascal.VOCSegmentation(split='train', transform=composed_transforms_tr)
        val_set = pascal.VOCSegmentation(split='val', transform=composed_transforms_val)
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
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

    else:
        raise NotImplementedError

