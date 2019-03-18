from modeling.deeplab import DeepLab
import torch
import torch.nn as nn
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from collections import OrderedDict


class SBoxOnXXX(nn.Module):
    def __init__(self, backbone='resnet'):
        super(SBoxOnXXX, self).__init__()
        self.base = DeepLab(backbone=backbone)
        self.last_conv = nn.Sequential(
            SynchronizedBatchNorm2d(self.base.num_classes),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(self.base.num_classes, self.base.num_classes // 2, 1, bias=False),
            SynchronizedBatchNorm2d(self.base.num_classes // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(self.base.num_classes // 2, 2, 1)
        )
        self._init_weights()

    def forward(self, x):
        x, pfm, low_level_feat = self.base(x)
        x = self.last_conv(x)
        return x, pfm, low_level_feat

    def load(self, path):
        print("Initializing weights from: {} for base network".format(path))
        # state_dict_checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        checkpoint = torch.load(path)

        # Remove the prefix .module from the model when it is trained using DataParallel
        if 'module.' in list(checkpoint['state_dict'].keys())[0]:
            print("test")
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict']:
                name = k[7:]  # remove `module.` from multi-gpu training
                new_state_dict[name] = v
        else:
            new_state_dict = checkpoint['state_dict']
        self.base.load_state_dict(new_state_dict)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def train_mode(self):
        self.base.train_mode()


if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    fms = SBoxOnXXX()
    fms.eval()
    result = fms(x)
    for i in result:
        print(i.size())
