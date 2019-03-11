import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ClickNet(nn.Module):
    def __init__(self):
        super(ClickNet, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(4, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.refine = nn.Sequential(
            nn.Conv2d(320, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 2, 3, 1, 1, bias=False),
        )

    def forward(self, input):
        x = torch.cat([input['pred_origin'], input['pos_map'], input['neg_map']], dim=1)
        x = self.downsample(x)
        x1 = torch.cat([input['fpm'], x], dim=1)
        result = self.refine(x1)
        result = F.interpolate(result, 256, mode='bilinear')
        return result + input['pred_origin']


if __name__ == '__main__':
    sample = {}
    sample['pred_origin'] = torch.randn(4, 2, 256, 256)
    sample['pos_map'] = torch.randn(4, 1, 256, 256)
    sample['neg_map'] = torch.randn(4, 1, 256, 256)
    sample['fpm'] = torch.randn(4, 256, 64, 64)
    click = ClickNet()
    print(click(sample).size())
