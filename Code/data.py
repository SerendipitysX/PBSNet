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

import torch.utils.data as data
import torch
import h5py
import cv2
import numpy as np
import scipy.io as sio


def get_edge(data):  # for training: NxHxWxC
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            # rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5))
            rs[i, :, :] = data[i, :, :] - cv2.GaussianBlur(data[i, :, :], (5,5), sigmaX=3,sigmaY=3)
        else:
            # rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
            rs[i, :, :, :] = data[i, :, :, :] - cv2.GaussianBlur(data[i, :, :, :],  (5,5), sigmaX=3,sigmaY=3)

    return rs



class Dataset_Pro(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro, self).__init__()
        #data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3=8806x8x64x64# for small data (not v7.3 data)
        # loading data
        data = h5py.File(file_path)

        # tensor type:NxHxWxC
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / 2047.
        gt1 = np.swapaxes(gt1, 1, 3)###NxCxWxH
        gt1 = np.swapaxes(gt1, 2, 3)  ###NxCxHxW
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:

        print(self.gt.size())

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / 2047.
        lms1 = np.swapaxes(lms1, 1, 3)  ###NxCxWxH
        lms1 = np.swapaxes(lms1, 2, 3)  ###NxCxHxW
        self.lms = torch.from_numpy(lms1)

        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / 2047.
        pan1 = get_edge(pan1)
        pan1 = np.expand_dims(pan1, axis=1)
        self.pan = torch.from_numpy(pan1) # Nx1xHxW:

        ms1 = data["ms"][...]  # NxHxWxC=0,1,2,3
        ms1 = np.array(ms1, dtype=np.float32) / 2047.  # NxHxWxC
        ms1 = get_edge(ms1)
        ms1 = np.array(ms1.transpose(0, 3, 1, 2))
        self.ms = torch.from_numpy(ms1)  # NxCxHxW:

    def __getitem__(self, index):
        return self.gt[index, :, :, :].float(), \
               self.lms[index, :, :, :].float(), \
               self.ms[index, :, :, :].float(), \
               self.pan[index, :, :, :].float()

    def __len__(self):
        return self.gt.shape[0]
