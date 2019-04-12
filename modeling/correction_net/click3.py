"""
Version 3 of click net, fuse user interactive with high-level features at first, then low-level feature to enhance
the details.
Next: object select click net VS details refine click net
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ClickNet(nn.Module):
    def __init__(self):
        super(ClickNet, self).__init__()
        self.reduce_c_aspp = nn.Sequential(
            nn.Conv2d(256, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.refine = nn.Sequential(
            nn.Conv2d(322, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, 2, 3, 1, 1),
        )
        self._init_weight()

    def forward(self, gaussian_center_map, aspp_feat, low_level_feat):
        low_feat = F.interpolate(low_level_feat, gaussian_center_map.size()[2:], mode='bilinear', align_corners=True)
        aspp_feat = F.interpolate(aspp_feat, gaussian_center_map.size()[2:], mode='bilinear', align_corners=True)
        aspp_feat = self.reduce_c_aspp(aspp_feat)
        x1 = torch.cat([low_feat, aspp_feat, gaussian_center_map], dim=1)
        result1 = self.refine(x1)
        result1 = F.interpolate(result1, gaussian_center_map.size()[2:], mode='bilinear', align_corners=True)
        return result1

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == '__main__':
    # image = torch.randn(2, 3, 512, 512)
    fused = torch.randn(2, 2, 256, 256)
    fpm = torch.randn(2, 256, 16, 16)
    low = torch.randn(2, 256, 64, 64)
    click = ClickNet()
    print(click(fused, fpm, low).size())
