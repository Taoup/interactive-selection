from modeling.correction_net.sbox_net import *
from modeling.correction_net.click_net import *
import torch.nn as nn
import numpy as np
from scipy import ndimage

DEBUG = False

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
        if 'crop_gt' in input.keys():  # training or validation
            pos_map, neg_map = self._simulate_user_interaction(sbox_pred, input['crop_gt'])
        else:  # inference with real user interaction
            user_clicked = False
            pos_clicks, neg_clicks = [], []
            if 'pos_clicks' in input:
                pos_clicks = input['pos_clicks']
                user_clicked = True
            if 'neg_clicks' in input:
                neg_clicks = input['neg_clicks']
                user_clicked = True
            if not user_clicked:
                # no user clicks, return result of surrounding box net directly
                return sbox_pred
            pos_map = self.convert_clicks(pos_clicks, image.shape[2:])
            neg_map = self.convert_clicks(neg_clicks, image.shape[2:])
        if image.is_cuda:
            pos_map, neg_map = pos_map.cuda(), neg_map.cuda()
        edm = torch.cat([pos_map, neg_map], dim=1)
        click_pred = self.click_net(sbox_pred, edm, fpm)
        return click_pred if not DEBUG else click_pred, pos_map, neg_map

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

    def convert_clicks(self, clicks, shape):
        gt = np.zeros(shape, dtype=np.float64)
        gt[...] = 255
        for click in clicks:
            xs = np.arange(0, shape[1], 1, np.float)
            ys = np.arange(0, shape[0], 1, np.float)
            ys = ys[:, np.newaxis]

            euclid = np.sqrt((xs - click[1]) ** 2 + (ys - click[0]) ** 2)
            euclid[euclid > 255] = 255
            gt = np.minimum(gt, euclid)
        gt = 1 - (gt.astype(np.float32) / 255)[np.newaxis, np.newaxis, ...]
        return torch.from_numpy(gt)


if __name__ == '__main__':
    DEBUG = True
    fusion = FusionNet(sbox='../../run/model_best.pth.tar').cuda()
    image = torch.randn((1, 3, 256, 256)).cuda()
    gt = torch.randn((1, 256, 256)).cuda()
    result, pos, neg = fusion(image, neg_clicks=[(40, 40), (100, 200)])
    print(result.size())
    import matplotlib.pyplot as plt

    plt.imshow(neg[0][0], cmap='gray')
    plt.show()
