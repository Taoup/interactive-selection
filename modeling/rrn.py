from modeling.fpbackbone import Bottleneck, FPbackbone
from modeling.sync_batchnorm import SynchronizedBatchNorm2d

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(258, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 32, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 16, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 4, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(4)
        self.conv5 = nn.Conv2d(4, 2, 3, 1, 1, bias=False)

    def forward(self, x, pos_map, neg_map):
        cat = torch.cat((x, pos_map, neg_map), 1)
        out1 = F.relu(self.bn1(self.conv1(cat)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))
        out4 = F.relu(self.bn4(self.conv4(out3)))
        out5 = self.conv5(out4)

        return out5


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(260, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 4, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(4)
        self.conv4 = nn.Conv2d(4, 2, 3, 1, 1, bias=False)

    def forward(self, prev_pred, x, pos_map, neg_map):
        cat = torch.cat((prev_pred, x, pos_map, neg_map), 1)
        out1 = F.relu(self.bn1(self.conv1(cat)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))
        out4 = self.conv4(out3)

        return out4 + prev_pred


class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(260, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 2, 3, 1, 1, bias=False)

    def forward(self, prev_pred, x, pos_map, neg_map):
        cat = torch.cat((prev_pred, x, pos_map, neg_map), 1)
        out1 = F.relu(self.bn1(self.conv1(cat)))
        out2 = self.conv2(out1)

        return out2 + prev_pred


class RRN(nn.Module):
    def __init__(self):
        super(RRN, self).__init__()
        self.backbone = FPbackbone(Bottleneck, [2, 2, 2, 2])
        self.net1 = Net1()
        self.net2 = Net2()
        self.net3 = Net3()
        self._init_weight()

    def forward(self, x, pos, neg):
        p1, p2, p3 = self.backbone(x)
        pos1 = F.interpolate(pos, scale_factor=0.25)
        neg1 = F.interpolate(neg, scale_factor=0.25)
        pred3 = self.net1(p3, pos1, neg1)  # 75*75
        pos2 = F.interpolate(pos, scale_factor=0.5)
        neg2 = F.interpolate(neg, scale_factor=0.5)
        pred1_upscaled = F.interpolate(pred3, scale_factor=2)
        pred2 = self.net2(pred1_upscaled, p2, pos2, neg2)  # 150*150
        pred2_upscaled = F.interpolate(pred2, scale_factor=2)
        pred1 = self.net3(pred2_upscaled, p1, pos, neg)  # 300*300
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
    net = RRN()
    out = net(x, pos, neg)
    for x in out:
        print(x.size())
