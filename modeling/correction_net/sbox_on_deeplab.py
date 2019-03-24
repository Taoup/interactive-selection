from modeling.deeplab import DeepLab
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from collections import OrderedDict


class SBoxOnDeeplab(nn.Module):
    def __init__(self, backbone='resnet', not_fine_tune=()):
        super(SBoxOnDeeplab, self).__init__()
        self.base = DeepLab(backbone=backbone)
        self.last_conv = nn.Sequential(
            SynchronizedBatchNorm2d(self.base.num_classes),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(self.base.num_classes, self.base.num_classes, 1, bias=False),
            SynchronizedBatchNorm2d(self.base.num_classes),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(self.base.num_classes, 2, 1)
        )
        if not isinstance(not_fine_tune, tuple):
            self.not_fine_tune = tuple(not_fine_tune)
        else:
            self.not_fine_tune = not_fine_tune
        for module in self.not_fine_tune:
            for p in getattr(self.base, module).parameters():
                p.requires_grad = False
        self._init_weights()

    def forward(self, image):
        x, pfm, low_level_feat = self.base(image)
        x = self.last_conv(x)
        x = F.interpolate(x, size=image.size()[2:], mode='bilinear', align_corners=True)
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
        model_dict = self.base.state_dict()
        new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
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

    def rest_parameters(self):
        modules = [self.last_conv]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def deeplab_parameters(self):
        modules = [self.base.backbone, self.base.decoder, self.base.aspp]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p



if __name__ == '__main__':
    x = torch.randn(1, 3, 512, 512)
    fms = SBoxOnDeeplab()
    fms.eval()
    result = fms(x)
    for i in result:
        print(i.size())
