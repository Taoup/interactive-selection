import os
import torch
import cv2
from collections import OrderedDict
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

from torch.nn.functional import upsample
from modeling.deeplab import *
from dataloaders import helpers as helpers
from dataloaders.custom_transforms import SimUserInput

pad = 50
thres = 0.8
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")

net = DeepLab(num_classes=2,
              backbone='xception',
              output_stride=16,
              sync_bn=False,
              freeze_bn=True)
#  Create the network and load the weights
model_dir = './run/pascal/deeplab-xception'
model_path = os.path.join(model_dir, 'checkpoint_1.pth.tar')
print("Initializing weights from: {}".format(model_path))
# state_dict_checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
checkpoint = torch.load(model_path)

# Remove the prefix .module from the model when it is trained using DataParallel
if 'module.' in list(checkpoint['state_dict'].keys())[0]:
    print("test")
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict']:
        name = k[7:]  # remove `module.` from multi-gpu training
        new_state_dict[name] = v
else:
    new_state_dict = checkpoint['state_dict']
net.load_state_dict(new_state_dict)
net.eval()
net.to(device)

#  Read image and click the points
rect_drawed = False
pos_points = []
neg_points = []
rect = []

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

tr = SimUserInput()
im_disp = image.copy()
cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_cb)
prev_total_len = -1
with torch.no_grad():
    while 1:
        cv2.imshow('image', im_disp)
        key = cv2.waitKey(1)
        if key & 0xff == ord('q'):
            exit(0)
        elif key & 0xff == ord(' '):
            im_disp = image.copy()
            rect = []
            pos_points, neg_points = [], []
            rect_drawed = False
            cv2.imshow('image', im_disp)
            cv2.destroyWindow('result')
            prev_total_len = -1

        if len(rect) == 2 and prev_total_len != (len(pos_points) + len(neg_points)):
            prev_total_len = len(pos_points) + len(neg_points)

            crop_image = image[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0], :]
            pos_map = tr.gen_EDM(pos_points, crop_image.shape[:2], 60)
            neg_map = tr.gen_EDM(neg_points, crop_image.shape[:2], 60)
            # cv2.imshow('pos map', pos_map)
            # cv2.imshow('neg map', neg_map)
            resize_image = helpers.fixed_resize(crop_image, (512, 512)).astype(np.float32)
            pos_map = helpers.fixed_resize(pos_map, (512, 512)).astype(np.float32)
            neg_map = helpers.fixed_resize(neg_map, (512, 512)).astype(np.float32)
            #  Concatenate inputs and convert to tensor
            concat = np.concatenate((resize_image, neg_map[..., np.newaxis]), axis=2)
            concat = np.concatenate((concat, pos_map[..., np.newaxis]), axis=2).astype(np.float32)
            concat = concat / 255
            inputs = torch.from_numpy(concat.transpose((2, 0, 1))[np.newaxis, ...])

            # Run a forward pass
            inputs = inputs.to(device)
            outputs = net.forward(inputs)
            outputs = outputs.to(torch.device('cpu'))

            pred = np.transpose(outputs.data.numpy()[0, ...], (1, 2, 0))
            pred = np.argmax(pred, axis=2)
            shape = crop_image.shape[:2]
            pred = cv2.resize(pred, shape[::-1], interpolation=cv2.INTER_NEAREST)
            show_image = crop_image.copy()
            show_image[..., 2] = show_image[..., 2] + pred * 255
            cv2.imshow('result', show_image)

