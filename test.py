import os
import torch
import cv2
from collections import OrderedDict
from PIL import Image
import numpy as np

from modeling.correction_net.sbox_net import *
from modeling.correction_net.sbox_on_deeplab import *
from modeling.correction_net.fusion_net import *
from modeling.correction_net.click5 import ClickNet
from modeling.deeplab1 import DeepLabX
from dataloaders import helpers as helpers
from dataloaders import custom_transforms as tr
from torchvision import transforms
import torch.nn.functional as F

pad = 50
thres = 0.8
gpu_id = 0
device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

wrapper_net = FusionNet(DeepLabX(pretrain=False), ClickNet())
wrapper_net.load_state_dict(torch.load('run/fusion_8907.pth.tar')['state_dict'])
# wrapper_net.sbox_net.load_state_dict(torch.load('run/sbox/sbox_miou_8735.pth.tar', map_location=device)['state_dict'])
wrapper_net.eval()
wrapper_net = wrapper_net.to(device)
wrapper_net.prev_pred = None

#  Read image and click the points
rect_drawed = False
pos_points = []
neg_points = []
rect = []
clicked = False

image = np.array(Image.open('ims/dog-cat.jpg'))

user_interaction = tr.SimUserInput()
test_transformer = transforms.Compose([
    tr.Normalize(),
])


def pred2cmap(pred):
    final_pred = pred.data.cpu()
    pred_score = F.softmax(final_pred)[0][1].numpy()
    cmap = cv2.applyColorMap((pred_score * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cmap


crop_image = image[0:400, 0:400, :]
resize_image = helpers.fixed_resize(crop_image, (256, 256)).astype(np.float32)
sample = {}
sample['crop_image'] = resize_image
sample = test_transformer(sample)
inputs = torch.from_numpy(sample['crop_image'].transpose((2, 0, 1))[np.newaxis, ...])

# Run a forward pass
with torch.no_grad():
    inputs = inputs.to(device)
    sbox_pred, fused_feat_maps, low_feat = wrapper_net.sbox_net(inputs)
    sbox_pred = F.interpolate(sbox_pred, crop_image.shape[:2], align_corners=True, mode='bilinear')
pred_cmap = pred2cmap(sbox_pred)
cv2.imwrite('ims/cmap.png', pred_cmap)
