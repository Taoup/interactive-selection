import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ClickNet(nn.Module):
    def __init__(self):
        super(ClickNet, self).__init__()
        self.reduce_c_low = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.reduce_c_aspp = nn.Sequential(
            nn.Conv2d(256, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.refine = nn.Sequential(
            nn.Conv2d(98, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(32, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(8, 2, 3, 1, 1),
        )

    def forward(self, pred, aspp_out, low_level_feat):
        down_pred = F.interpolate(pred, low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        aspp_out = F.interpolate(aspp_out, low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        aspp_out = self.reduce_c_aspp(aspp_out)
        low_level_feat = self.reduce_c_low(low_level_feat)
        x1 = torch.cat([down_pred, aspp_out, low_level_feat], dim=1)
        result1 = self.refine(x1)
        result1 = F.interpolate(result1, pred.size()[2:], mode='bilinear', align_corners=True)
        return result1


if __name__ == '__main__':
    fused = torch.randn(2, 2, 256, 256)
    fpm = torch.randn(2, 256, 16, 16)
    low = torch.randn(2, 256, 64, 64)
    click = ClickNet()
    print(click(fused, fpm, low).size())
