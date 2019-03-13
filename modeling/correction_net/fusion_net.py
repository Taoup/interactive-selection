from modeling.correction_net.sbox_net import *
from modeling.correction_net.click_net import *
import torch.nn as nn
import numpy as np
from scipy import ndimage


class FusionNet(nn.Module):
    def __init__(self, sbox=None, click=None):
        super(FusionNet, self).__init__()
        self.sbox_net = SBoxNet()
        self.click_net = ClickNet()
        if sbox:
            self.sbox_net.load(sbox)
        if click:
            self.click_net.load(click)

    def forward(self, image, **input):
        sbox_pred, fpm = self.sbox_net(image)
        if 'crop_gt' in input.keys():
            pos_map, neg_map = self._simulate_user_interaction(sbox_pred, input['crop_gt'])
            if image.is_cuda:
                pos_map, neg_map = pos_map.cuda(), neg_map.cuda()
        else:
            return sbox_pred
            pass
        edm = torch.cat([pos_map, neg_map], dim=1)
        click_pred = self.click_net(sbox_pred, edm, fpm)
        return click_pred

    def _simulate_user_interaction(self, pred, gt):
        def __gen_EDM(mask):
            edm_list = []
            for i in range(mask.shape[0]):
                tmp = mask[i].cpu()
                tmp = ndimage.binary_erosion(tmp, structure=np.ones((5, 5))).astype(np.uint8)
                label, num = ndimage.label(tmp)
                shape = tmp.shape
                udm = np.zeros(shape, dtype=np.float32)
                if num != 0:
                    max_label = np.argmax(ndimage.sum(tmp, label, range(1, num + 1))) + 1
                    idx_xs, idx_ys = np.where(label == max_label)
                    _tmp = np.random.randint(0, len(idx_xs))
                    xs = np.arange(0, shape[1], 1, np.float)
                    ys = np.arange(0, shape[0], 1, np.float)
                    ys = ys[:, np.newaxis]
                    udm = np.sqrt((xs - idx_xs[_tmp]) ** 2 + (ys - idx_ys[_tmp]) ** 2).astype(np.float32)
                    udm[udm > 255] = 255
                    udm = (255 - udm) / 255
                edm_list.append(torch.from_numpy(udm[np.newaxis, np.newaxis, ...]))

            return torch.cat(edm_list, dim=0)

        _pred = torch.argmax(pred, dim=1).float()
        FPs = (_pred > gt)
        FNs = (_pred < gt)
        neg_map = __gen_EDM(FPs)
        pos_map = __gen_EDM(FNs)
        return pos_map, neg_map


if __name__ == '__main__':
    fusion = FusionNet(sbox='../../run/model_best.pth.tar').cuda()
    image = torch.randn((2, 3, 256, 256)).cuda()
    gt = torch.randn((2, 256, 256)).cuda()
    result = fusion(image, crop_gt=gt)
    print(result.size())
