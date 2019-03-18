from modeling.fpn import FPN
from modeling.fpbackbone import Bottleneck, FPbackbone

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1_32 = nn.Conv2d(256, 128, 3, 1, 1, bias=False)
        self.bn1_32 = nn.BatchNorm2d(128)
        self.conv2_32 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.bn2_32 = nn.BatchNorm2d(128)
        self.conv3_32 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.bn3_32 = nn.BatchNorm2d(128)

        self.conv1_16 = nn.Conv2d(256, 128, 3, 1, 1, bias=False)
        self.bn1_16 = nn.BatchNorm2d(128)
        self.conv2_16 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.bn2_16 = nn.BatchNorm2d(128)

        self.conv1_8 = nn.Conv2d(256, 128, 3, 1, 1, bias=False)
        self.bn1_8 = nn.BatchNorm2d(128)

        self.conv1_4 = nn.Conv2d(256, 128, 3, 1, 1, bias=False)
        self.bn1_4 = nn.BatchNorm2d(128)

        self.conv_final = nn.Conv2d(128, 2, 3, 1, 1)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, p1, p2, p3, p4):
        out1 = F.relu(self.bn1_32(self.conv1_32(p4)))
        out1 = F.interpolate(out1, scale_factor=2, mode='bilinear')
        out1 = F.relu(self.bn2_32(self.conv2_32(out1)))
        out1 = F.interpolate(out1, scale_factor=2, mode='bilinear')
        out1 = F.relu(self.bn3_32(self.conv3_32(out1)))
        out1 = F.interpolate(out1, scale_factor=2, mode='bilinear')

        out2 = F.relu(self.bn1_16(self.conv1_16(p3)))
        out2 = F.interpolate(out2, scale_factor=2, mode='bilinear')
        out2 = F.relu(self.bn2_16(self.conv2_16(out2)))
        out2 = F.interpolate(out2, scale_factor=2, mode='bilinear')

        out3 = F.relu(self.bn1_8(self.conv1_8(p2)))
        out3 = F.interpolate(out3, scale_factor=2, mode='bilinear')

        out4 = F.relu(self.bn1_4(self.conv1_4(p1)))
        fused_pfm = out1 + out2 + out3 + out4
        result = self.conv_final(fused_pfm)
        result = F.interpolate(result, scale_factor=4, mode='bilinear')

        return result, fused_pfm


class SBoxNet(nn.Module):
    def __init__(self):
        super(SBoxNet, self).__init__()
        self.backbone = FPN(Bottleneck, [2, 4, 23, 3], pretrained=True)
        self.decoder = Decoder()

    def forward(self, x):
        p1, p2, p3, p4 = self.backbone(x)
        result, fused_pfm = self.decoder(p1, p2, p3, p4)
        return result, fused_pfm

    def load(self, path):
        print("Initializing weights from: {}".format(path))
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
        self.load_state_dict(new_state_dict)

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == '__main__':
    x = Variable(torch.randn(5, 3, 256, 256))
    fms = SBoxNet()
    result, fpm = fms(x)
    print(result.size())
    print(fpm.size())
