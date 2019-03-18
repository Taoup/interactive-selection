import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ClickNet(nn.Module):
    def __init__(self):
        super(ClickNet, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(2, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(192, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(32, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(8, 2, 3, 1, 1),
        )

    def forward(self, pred, fpm):
        x = self.downsample(pred)
        fpm = F.interpolate(fpm, x.size()[2:], mode='bilinear', align_corners=True)
        x1 = torch.cat([fpm, x], dim=1)
        result1 = self.refine(x1)
        result1 = F.interpolate(result1, pred.size()[2:], mode='bilinear', align_corners=True)
        return result1


if __name__ == '__main__':
    fused = torch.randn(2, 2, 256, 256)
    fpm = torch.randn(2, 256, 16, 16)
    low = torch.randn(2, 256, 64, 64)
    click = ClickNet()
    print(click(fused, fpm, low).size())
