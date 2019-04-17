import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone
from collections import OrderedDict


class DeepLabX(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False, pretrain=True):
        super(DeepLabX, self).__init__()
        self.num_classes = num_classes
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

        if pretrain:
            self._load_pretrain()

        # change the last inference layer for binary segmentation mask
        last_conv = list(self.decoder.last_conv.children())
        self.decoder.last_conv = nn.Sequential(*last_conv[:-1])
        self.decoder.last_conv.add_module('8', nn.Conv2d(256, 2, 1, 1))

    def _load_pretrain(self):
        path = 'run/resnet/deeplab-resnet.pth.tar'
        print("load pretrained Deeplab from: {} ".format(path))
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
        model_dict = self.state_dict()
        new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        self.load_state_dict(new_state_dict)

    def forward(self, input):
        x, low_pooled, low_level_feat = self.backbone(input)
        aspp_feat = self.aspp(x)
        x = self.decoder(aspp_feat, low_pooled)
        # x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x, aspp_feat, low_level_feat

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = DeepLabX(backbone='resnet', output_stride=16, pretrain=False)
    model.eval()
    input = torch.rand(1, 3, 512, 512)
    output = model(input)
    for x in output:
        print(x.size())
