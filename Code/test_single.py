# ---------------------------------------------------------------
# Copyright (c) 2021, Shishi Xiao, Cheng Jin, Tian-Jing Zhang, 
# Ran Ran, Liang-Jian Deng, All rights reserved.
#
# This work is licensed under GNU Affero General Public License 
# v3.0 International To view a copy of this license, see the 
# LICENSE file.
#
# This file is running on WorldView-3 dataset. For other dataset
# (i.e., QuickBird and GaoFen-2), please change the corresponding
# inputs.
# ---------------------------------------------------------------

import torch.nn.modules as nn
import torch
import cv2
import numpy as np
from model_separate_v1 import OURNet
import h5py
import scipy.io as sio
import os

###################################################################
# ------------------- Sub-Functions (will be used) -------------------
###################################################################
def get_edge(data):  # for training: HxWxC
    rs = np.zeros_like(data)
    if len(rs.shape) == 3:
        for i in range(data.shape[2]):
            rs[:, :, i] = data[:, :, i] - cv2.GaussianBlur(data[:, :, i], (5,5), sigmaX=3,sigmaY=3)
    else:
        rs = data - cv2.GaussianBlur(data, (5,5), sigmaX=3,sigmaY=3)
    return rs



def load_set(file_path):
    data = sio.loadmat(file_path)  # HxWxC=256x256x8

    # tensor type:
    lms = torch.from_numpy(data['lms'] / 2047.0).permute(2, 0, 1)  # CxHxW = 8x256x256
    ms = torch.from_numpy(get_edge(data['ms'] )/ 2047.0).permute(2, 0, 1)  # CxHxW= 8x64x64
    pan = torch.from_numpy(get_edge(data['pan'] / 2047.0))  # HxW = 256x256

    return lms, ms, pan

def load_gt_compared(file_path):
    data = sio.loadmat(file_path)  # HxWxC=256x256x8

    # tensor type:
    test_gt = torch.from_numpy(data['gt'] / 2047.0)  # CxHxW = 8x256x256

    return test_gt

###################################################################
# ------------------- Main Test (Run second) -------------------
###################################################################
ckpt = "Weights/390.pth"   # chose model

def test(file_path):
    lms, ms, pan = load_set(file_path)

    model = OURNet().cuda().eval()   # fixed, important!

    weight = torch.load(ckpt)  # load Weights!

    model.load_state_dict(weight) # fixed

    with torch.no_grad():

        x1, x2, x3 = lms, ms, pan   # read data: CxHxW (numpy type)
        print(x1.shape)
        x1 = x1.cuda().unsqueeze(dim=0).float()  # convert to tensor type: 1xCxHxW (unsqueeze(dim=0))
        x2 = x2.cuda().unsqueeze(dim=0).float()  # convert to tensor type: 1xCxHxW (unsqueeze(dim=0))
        x3 = x3.cuda().unsqueeze(dim=0).unsqueeze(dim=1).float()  # convert to tensor type: 1x1xHxW

        output1, output2, output3 = model(x2, x3, x1)  # call model
        # tensor type: CxHxW

        # convert to numpy type with permute and squeeze: HxWxC (go to cpu for easy saving)
        sr = torch.squeeze(output3).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC

        print(sr.shape)
        save_name = os.path.join("results", "psfn.mat")  # fixed! save as .mat format that will used in Matlab!
        sio.savemat(save_name, {'psfn': sr})  # fixed!

###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == '__main__':
    file_path = "/path/to/test/data" # put test data here
    test(file_path)   # recall test function
