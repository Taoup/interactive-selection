class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return 'D:\\datasets\\'
        elif dataset == 'tmall':
            return 'D:\\datasets\\Tmall'
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'grabcut':
            return 'D:\\datasets\\grab_cut_history'
        elif dataset == 'bekeley':
            return 'D:\\datasets\\bekeley_100'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
