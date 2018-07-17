import torch
import torch.nn as nn
from torch.utils.serialization import load_lua
from PIL import Image
import numpy as np
import cv2
import os
import segmentation_evaluation
import unet as un

on_gpu = True

def test_nyu_model(model_folder,depth=False):
    print('Evaluating folder:{}'.format(model_folder))
    unet = un.UNetRGBD(14) if depth else un.UNet(14)
    unet.load_state_dict(torch.load(os.path.join(model_folder,'converted_clean_model.pth')))
    if on_gpu:
        unet.cuda()
    unet.eval()
    preds = []
    gts = []
    for i in range(1,1450):
        img_id = str(i).zfill(4)
        try:
            scaled_rgb = np.load('nyu_data/{}_rgb.npy'.format(img_id))
            scaled_depth = np.load('nyu_data/{}_depth.npy'.format(img_id))
            gt = cv2.imread(os.path.join('nyu_data/13_class_labels/{}_label.png'.format(img_id)),cv2.IMREAD_ANYDEPTH)
        except:
            continue
        scaled_rgb = np.expand_dims(scaled_rgb,0)
        scaled_depth = np.expand_dims(scaled_depth,0)
        torch_rgb = torch.tensor(scaled_rgb,dtype=torch.float32)
        torch_depth = torch.tensor(scaled_depth,dtype=torch.float32)
        if on_gpu:
            if depth:
                pred = unet.forward((torch_rgb.cuda(),torch_depth.cuda()))
            else:
                pred = unet.forward(torch_rgb.cuda())
            pred_numpy = pred.cpu().detach().numpy()
        else:
            if depth:
                pred = unet.forward((torch_rgb,torch_depth))
            else:
                pred = unet.forward(torch_rgb)
            pred_numpy = pred.detach().numpy()
        new_pred = np.argmax(pred_numpy[0],axis=0)
        preds.append(new_pred)
        gts.append(gt)
    segmentation_evaluation.evaluate_lists(preds,gts)

def test_sun_model(model_folder,depth=False):
    print('Evaluating folder:{}'.format(model_folder))
    unet = un.UNetRGBD(14) if depth else un.UNet(14)
    unet.load_state_dict(torch.load(os.path.join(model_folder,'converted_clean_model.pth')))
    if on_gpu:
        unet.cuda()
    unet.eval()
    all_preds = []
    all_gt = []
    for i in range(1,5051):
        img_id = str(i).zfill(4)
        try:
            scaled_rgb = np.load('sunrgbd_data/{}_rgb.npy'.format(img_id))
            scaled_depth = np.load('sunrgbd_data/{}_depth.npy'.format(img_id))
            gt = cv2.imread(os.path.join('sunrgbd_data/13_class_labels/{}_label.png'.format(img_id)),cv2.IMREAD_ANYDEPTH)
        except:
            continue
        scaled_rgb = np.expand_dims(scaled_rgb,0)
        scaled_depth = np.expand_dims(scaled_depth,0)
        torch_rgb = torch.tensor(scaled_rgb,dtype=torch.float32)
        torch_depth = torch.tensor(scaled_depth,dtype=torch.float32)
        if on_gpu:
            if depth:
                pred = unet.forward((torch_rgb.cuda(),torch_depth.cuda()))
            else:
                pred = unet.forward(torch_rgb.cuda())
            pred_numpy = pred.cpu().detach().numpy()
        else:
            if depth:
                pred = unet.forward((torch_rgb,torch_depth))
            else:
                pred = unet.forward(torch_rgb)
            pred_numpy = pred.detach().numpy()
        new_pred = np.argmax(pred_numpy[0],axis=0)
        all_preds.append(new_pred)
        all_gt.append(gt)
    segmentation_evaluation.evaluate_lists(all_preds,all_gt)

test_nyu_model('nyu_models/rgb_no_pretrain',depth=False)
test_nyu_model('nyu_models/rgb_imagenet_pretrain',depth=False)
test_nyu_model('nyu_models/rgb_scenenet_pretrain',depth=False)
test_nyu_model('nyu_models/rgbd_no_pretrain',depth=True)
test_nyu_model('nyu_models/rgbd_scenenet_pretrain',depth=True)

test_sun_model('sun_models/rgb_no_pretrain',depth=False)
test_sun_model('sun_models/rgb_imagenet_pretrain',depth=False)
test_sun_model('sun_models/rgb_scenenet_pretrain',depth=False)
test_sun_model('sun_models/rgbd_no_pretrain',depth=True)
test_sun_model('sun_models/rgbd_scenenet_pretrain',depth=True)
