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
# d0 = 120
# n=1
#
# def fft_distances(m, n):
#     u = np.array([i if i <= m / 2 else m - i for i in range(m)],
#                  dtype=np.float32)
#     v = np.array([i if i <= n / 2 else n - i for i in range(n)],
#                  dtype=np.float32)
#     v.shape = n, 1
#     ret = np.sqrt(u * u + v * v)
#     return np.fft.fftshift(ret)
#
# ##butterworth # for training: NxHxWxC
# def get_edge(data):
#     rs = np.zeros_like(data)
#     m = data.shape[1]
#     n = data.shape[2]
#     N = data.shape[0]
#     for i in range(N):
#         if len(data.shape) ==3:#N H W
#             img = data[i,:,:]
#             img_float32 = np.float32(img)
#             dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)  # foriour
#             fft_mat = np.fft.fftshift(dft)  # get central low frequency
#             ###frequecy operation
#             duv = fft_distances(img.shape[0], img.shape[1])
#             crow, ccol = int(m / 2), int(n / 2)
#             duv[crow , ccol] = 0.000001
#             filter_mat = 1 / (1 + np.power(duv / d0, 2 * n))
#             filter_mat = cv2.merge((filter_mat, filter_mat))
#             img_idf = filter_mat * fft_mat
#             img_idf = np.fft.ifftshift(img_idf)
#             img_idf = cv2.idft(img_idf)
#             magnitude_spectrum = np.log(1 + cv2.magnitude(img_idf[:, :, 0], img_idf[:, :, 1]))  # real and imagine part --> spatial domain
#             rs[i,:,:]=magnitude_spectrum
#             cv2.normalize(magnitude_spectrum, magnitude_spectrum, 0, np.max(img_idf), cv2.NORM_MINMAX)
#         else:
#             for j in range(8):
#                 img = data[i, :, :, j]
#                 img_float32 = np.float32(img)
#                 dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)  # foriour
#                 fft_mat = np.fft.fftshift(dft)  # get central low frequency
#                 ###frequecy operation
#                 duv = fft_distances(img.shape[0], img.shape[1])
#                 crow, ccol = int(m / 2), int(n / 2)
#                 duv[crow, ccol] = 0.000001
#                 filter_mat = 1 / (1 + np.power(duv / d0, 2 * n))
#                 filter_mat = cv2.merge((filter_mat, filter_mat))
#                 img_idf = filter_mat * fft_mat
#                 img_idf = np.fft.ifftshift(img_idf)
#                 img_idf = cv2.idft(img_idf)
#                 magnitude_spectrum = np.log(
#                     1 + cv2.magnitude(img_idf[:, :, 0], img_idf[:, :, 1]))  # real and imagine part --> spatial domain
#                 rs[i, :, :, j] = magnitude_spectrum
#
#                 cv2.normalize(magnitude_spectrum, magnitude_spectrum, 0, np.max(img_idf), cv2.NORM_MINMAX)
#     return rs



class Dataset_Pro(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro, self).__init__()
        #######
        #data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3=8806x8x64x64# for small data (not v7.3 data)
        ############## loading data
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

    #####必要函数
    def __getitem__(self, index):
        return self.gt[index, :, :, :].float(), \
               self.lms[index, :, :, :].float(), \
               self.ms[index, :, :, :].float(), \
               self.pan[index, :, :, :].float()

            #####必要函数
    def __len__(self):
        return self.gt.shape[0]
