import os
import torch
import cv2
from collections import OrderedDict
from PIL import Image
import numpy as np

from modeling.correction_net.sbox_net import *
from modeling.correction_net.sbox_on_deeplab import *
from modeling.correction_net.fusion_net import *
from dataloaders import helpers as helpers
from dataloaders import custom_transforms as tr
from torchvision import transforms

pad = 50
thres = 0.8
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

wrapper_net = FusionNet(SBoxOnDeeplab(), ClickNet())
# wrapper_net.load_state_dict(torch.load('run/click/click_miou_8254.pth.tar')['state_dict'])
wrapper_net.sbox_net.load('run/sbox/sbox_miou_8102.pth.tar')
wrapper_net.eval()
wrapper_net = wrapper_net.to(device)
wrapper_net.prev_pred = None

#  Read image and click the points
rect_drawed = False
pos_points = []
neg_points = []
rect = []


def gaussian_clicks(clicks, shape, sigma=20):
    gt = np.zeros(shape, dtype=np.float64)
    for click in clicks:
        xs = np.arange(0, shape[1], 1, np.float)
        ys = np.arange(0, shape[0], 1, np.float)
        ys = ys[:, np.newaxis]

        euclid = np.exp(-4 * np.log(2) * ((xs - click[1]) ** 2 + (ys - click[0]) ** 2) / sigma ** 2)
        gt = np.maximum(euclid, gt)
    gt = gt.astype(np.float32)[np.newaxis, np.newaxis, ...]
    return torch.from_numpy(gt)

def mouse_cb(event, x, y, flag, para):
    global pos_points, neg_points, rect_drawed, rect

    if event == cv2.EVENT_LBUTTONDOWN:
        if rect_drawed:
            # Draw a green dot
            cv2.circle(im_disp, (x,y), 2, (0, 255, 0))
            pos_points.append((y - rect[0][1],x - rect[0][0]))
        else:
            rect.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        # draw a rectangle
        if not rect_drawed:
            rect.append((x, y))
            cv2.rectangle(im_disp,rect[0], rect[1], (0, 255, 0))
        rect_drawed = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        if rect_drawed:
            # draw a red dot
            cv2.circle(im_disp, (x,y), 2, (0, 0, 255))
            neg_points.append((y - rect[0][1],x - rect[0][0]))


image = np.array(Image.open('ims/bear.jpg'))

user_interaction = tr.SimUserInput()
test_transformer = transforms.Compose([
    tr.Normalize(),
])

im_disp = image.copy()
cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_cb)
prev_total_len = -1
with torch.no_grad():
    while 1:
        cv2.imshow('image', cv2.cvtColor(im_disp, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1)
        if key & 0xff == ord('q'):
            exit(0)
        elif key & 0xff == ord(' '):
            im_disp = image.copy()
            rect = []
            pos_points, neg_points = [], []
            rect_drawed = False
            wrapper_net.prev_pred = None
            cv2.imshow('image', im_disp)
            cv2.destroyWindow('result')
            prev_total_len = -1

        if len(rect) == 2 and prev_total_len != (len(pos_points) + len(neg_points)):
            prev_total_len = len(pos_points) + len(neg_points)
            sample = {}

            crop_image = image[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0], :]
            if wrapper_net.prev_pred is None:
                resize_image = helpers.fixed_resize(crop_image, (256, 256)).astype(np.float32)
                sample['crop_image'] = resize_image
                sample = test_transformer(sample)
                inputs = torch.from_numpy(sample['crop_image'].transpose((2, 0, 1))[np.newaxis, ...])

            # Run a forward pass
                inputs = inputs.to(device)
                pred, fused_feat_maps = wrapper_net.sbox_net(inputs)
                wrapper_net.prev_pred = pred
                wrapper_net.fused_feat_maps = fused_feat_maps
            else:
                pos_map = gaussian_clicks(pos_points, crop_image.shape[:2], 60)
                pos_map = F.interpolate(pos_map, (256, 256), mode='bilinear')
                neg_map = gaussian_clicks(neg_points, crop_image.shape[:2], 60)
                neg_map = F.interpolate(neg_map, (256, 256), mode='bilinear')
                gdm = torch.cat([neg_map, pos_map], dim=1).to(device)
                fused = wrapper_net.prev_pred + gdm
                pred = wrapper_net.click_net(fused, wrapper_net.fused_feat_maps)
                # wrapper_net.prev_pred = pred
            pred = pred.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)[0].astype(np.uint8)
            pred = cv2.resize(pred, tuple(reversed(crop_image.shape[:2])), interpolation=cv2.INTER_NEAREST)
            pred = pred * 255
            cv2.imshow('mask', pred)
            show_image = crop_image.copy()
            show_image[..., 0] = cv2.add(show_image[..., 0], pred)
            cv2.imshow('result', cv2.cvtColor(show_image, cv2.COLOR_RGB2BGR))
