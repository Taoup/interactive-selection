from modeling.fpbackbone import Bottleneck, FPbackbone
from modeling.sync_batchnorm import SynchronizedBatchNorm2d

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LocalizationNet(nn.Module):
    def __init__(self):
        super(LocalizationNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))
        out4 = F.relu(self.bn4(self.conv4(out3)))

        return out2, out4


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.conv1 = nn.Conv2d(68, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 4, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(4)
        self.conv4 = nn.Conv2d(4, 2, 3, 1, 1, bias=False)

    def forward(self, prev_pred, low_x, pos_map, neg_map):
        cat = torch.cat((prev_pred, low_x, pos_map, neg_map), 1)
        out1 = F.relu(self.bn1(self.conv1(cat)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))
        out4 = self.conv4(out3)

        return out4 + prev_pred


class PreliminaryNet(nn.Module):
    def __init__(self):
        super(PreliminaryNet, self).__init__()
        self.conv1 = nn.Conv2d(256, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 8, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 2, 3, 1, 1, bias=False)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = self.conv3(out2)
        return out3


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = nn.Conv2d(36, 8, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 2, 3, 1, 1, bias=False)

    def forward(self, prev_pred, x, pos_map, neg_map):
        cat = torch.cat((prev_pred, x, pos_map, neg_map), 1)
        out1 = F.relu(self.bn1(self.conv1(cat)))
        out2 = self.conv2(out1)

        return out2 + prev_pred


class ForkNet(nn.Module):
    def __init__(self):
        super(ForkNet, self).__init__()
        self.backbone = FPbackbone(Bottleneck, [2, 2, 2, 2])
        self.preliminary_net = PreliminaryNet()
        self.loc = LocalizationNet()
        self.fusion = FusionNet()
        self.refine = RefineNet()
        self._init_weight()

    def forward(self, x, pos, neg):
        p1 = self.backbone(x)
        pred1 = self.preliminary_net(p1)
        pred1_upscaled = F.interpolate(pred1, scale_factor=2)
        pos1 = F.interpolate(pos, scale_factor=0.5)
        neg1 = F.interpolate(neg, scale_factor=0.5)
        l1, l2 = self.loc(x)
        pred2 = self.fusion(pred1_upscaled, l2, pos1, neg1)
        pred2_upscaled = F.interpolate(pred2, scale_factor=2)
        pred3 = self.refine(pred2_upscaled, l1, pos, neg)
        return pred1, pred2, pred3

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == '__main__':
    from torch.autograd import Variable

    pos = Variable(torch.randn([4, 1, 300, 300]))
    neg = Variable(torch.randn([4, 1, 300, 300]))
    x = Variable(torch.randn([4, 3, 300, 300]))
    net = ForkNet()
    out = net(x, pos, neg)
    for x in out:
        print(x.size())
