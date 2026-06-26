# -*- coding: utf-8 -*-
"""
@author: caigentan@AnHui University
@software: PyCharm
@file: SwinNet_test.py
@time: 2021/5/27 09:34
"""

import time

import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from models.Swin_Transformer import SwinTransformer,SwinNet
from data import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--rgb', type=str, help="path to rgb images")
parser.add_argument('--depth', type=str, help="path to depth images")
parser.add_argument('--gt', type=str, help="path to gt images")
parser.add_argument('--target', type=str, help="path to save results")

opt = parser.parse_args()

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

#load the model
model = SwinNet()
model.load_state_dict(torch.load('./cpts/SwinTransNet_epoch_best.pth'))
model.cuda()
model.eval()

image_root = opt.rgb if opt.rgb.endswith('/') else opt.rgb + '/'
gt_root = opt.gt if opt.gt.endswith('/') else opt.gt + '/'
depth_root = opt.depth if opt.depth.endswith('/') else opt.depth + '/'
target_path = opt.target if opt.target.endswith('/') else opt.target + '/'
eval = target_path + 'evaluation.txt'

test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)

torch.cuda.synchronize()
start_time = time.time()

for i in range(test_loader.size):
    print(f'Processing image {i+1}/{test_loader.size}')
    image, gt, depth, name, image_for_post = test_loader.load_data()
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    image = image.cuda()
    depth = depth.repeat(1,3,1,1).cuda()

    res, edge = model(image,depth)
    res = F.interpolate(res, size=gt.shape[:2], mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)


    cv2.imwrite(target_path + name, (res*255).astype(np.uint8))

end_time = time.time()
total_time = end_time - start_time
images_processed = test_loader.size
average_time_per_image = total_time / images_processed if images_processed > 0 else 0

with open(eval, 'w') as f:
    f.write(f'Total Time: {total_time:.4f} seconds\n')
    f.write(f'Images Processed: {images_processed}\n')
    f.write(f'Average Time Per Image: {average_time_per_image:.4f} seconds\n')

print('Test Done!')
