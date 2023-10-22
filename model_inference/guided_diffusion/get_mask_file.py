import os
import cv2
import glob
import numpy as np


mask_route = '/root/autodl-tmp/colon_seg_normal/colon_seg_normal/train/fold'

new_route = '/root/autodl-tmp/dataset/mask'
if not os.path.exists(new_route):
    os.makedirs(new_route)
files = glob.glob(os.path.join(mask_route, '*.npy'))

for file in files:
    filename = os.path.basename(file).split('.npy')[0]

    mask = np.load(file)
    mask = 1 - mask

    img = mask * 255
    cv2.imwrite(os.path.join(new_route, filename+'.png'), img.astype(np.uint8))
