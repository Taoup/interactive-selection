from modeling.fpn import FPN
from modeling.fpbackbone import Bottleneck, FPbackbone

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BboxNet(nn.Module):
    def __init__(self):
        super(BboxNet, self).__init__()
        self.backbone = FPN(Bottleneck, [2, 2, 2, 2])
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

    def forward(self, x):
        p1, p2, p3, p4 = self.backbone(x)
        out1 = F.relu(self.bn1_32(self.conv1_32(p4)))
        out1 = F.upsample(out1, scale_factor=2, mode='bilinear')
        out1 = F.relu(self.bn2_32(self.conv2_32(out1)))
        out1 = F.upsample(out1, scale_factor=2, mode='bilinear')
        out1 = F.relu(self.bn3_32(self.conv3_32(out1)))
        out1 = F.upsample(out1, scale_factor=2, mode='bilinear')

        out2 = F.relu(self.bn1_16(self.conv1_16(p3)))
        out2 = F.upsample(out2, scale_factor=2, mode='bilinear')
        out2 = F.relu(self.bn2_16(self.conv2_16(out2)))
        out2 = F.upsample(out2, scale_factor=2, mode='bilinear')

        out3 = F.relu(self.bn1_8(self.conv1_8(p2)))
        out3 = F.upsample(out3, scale_factor=2, mode='bilinear')

        out4 = F.relu(self.bn1_4(self.conv1_4(p1)))
        pfm = out1 + out2 + out3 + out4
        result = self.conv_final(pfm)
        result = F.upsample(result, scale_factor=4, mode='bilinear')

        return result


if __name__ == '__main__':
    x = Variable(torch.randn(5, 3, 256, 256))
    fms = BboxNet()
    result = fms(x)
    print(result.size())
