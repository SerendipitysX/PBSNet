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

import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from evaluate import compute_index
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import Dataset_Pro
from model_separate_v1 import PBSNet, summaries, Resblock
from main_test_single import load_set, load_gt_compared
import numpy as np
import scipy.io as sio
from evaluate import compute_index
import shutil
from torch.utils.tensorboard import SummaryWriter

###################################################################
# ------------------- Pre-Define Part----------------------
###################################################################
# ============= 1) Pre-Define =================== #
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True
cudnn.benchmark = False

# ============= 2) HYPER PARAMS(Pre-Defined) ==========#
lr = 0.001
epochs = 200  # 450
ckpt = 50
batch_size = 32
model_path = "Weights/.pth"


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ============= 3) Load Model + Loss + Optimizer + Learn_rate_update ==========#
model = PBSNet().cuda()
if os.path.isfile(model_path):
    model.load_state_dict(torch.load(model_path))  ## Load the pretrained Encoder
    print('OURNet is Successfully Loaded from %s' % (model_path))

# summaries(model, grad=True)  ## Summary the Network
criterion = nn.MSELoss(size_average=True).cuda()  ## Define the Loss function


optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)  ## optimizer 1: Adam
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)   # learning-rate update

# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-7)  ## optimizer 2: SGD
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=180, gamma=0.1)  # learning-rate update: lr = lr* 1/gamma for each step_size = 180

# ============= 4) Tensorboard_show + Save_model ==========#
# if os.path.exists('train_logs'):  # for tensorboard: copy dir of train_logs  ## Tensorboard_show: case 1
#     shutil.rmtree('train_logs')  # ---> console (see tensorboard): tensorboard --logdir = dir of train_logs
if os.path.exists('./train_logs/50'):
    shutil.rmtree('./train_logs/50')
writer = SummaryWriter('./train_logs/50')  ## Tensorboard_show: case 2


def save_checkpoint(model, epoch):  # save model function
    model_out_path = 'Weights' + '/' + "{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)


###################################################################
# ------------------- Main Train (Run second)----------------------
###################################################################
def train(training_data_loader, validate_data_loader,test_lms,test_ms, test_pan, test_gt, start_epoch=0):
    print('Start training...')
    # epoch 450, 450*550 / 2 = 123750 / 8806 = 14/per imgage

    for epoch in range(start_epoch, epochs, 1):

        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []
        epoch_train_loss1, epoch_train_loss2, epoch_train_loss3, epoch_train_loss4 = [], [], [], []
        epoch_train_loss5, epoch_train_loss6, epoch_train_loss7, epoch_train_loss8 = [], [], [], []
        epoch_train_batch1, epoch_train_batch2 = [], []
        epoch_val_batch1, epoch_val_batch2, epoch_val_batch3 = [], [], []
        #####experiment
        epoch_train_loss1_1, epoch_train_loss2_1, epoch_train_loss3_2, epoch_train_loss4_2 = [], [], [], []
        epoch_train_loss5_2, epoch_train_loss6_2, epoch_train_loss8_2 = [], [], []


        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):
            # gt Nx8x64x64
            # lms Nx8x64x64
            # ms_hp Nx8x16x16
            # pan_hp Nx1x64x64
            gt, lms, ms, pan = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
            # print(gt.size())
            # print(lms.size())
            # print(ms.size())
            # print(pan.size())
            optimizer.zero_grad()  # fixed

            output1, output2, output3 = model(ms, pan,lms)  # call model

            #loss = criterion(output, gt)
            #epoch_train_loss.append(loss.item())

            # compute loss

            loss_band1 = criterion(output3[:, 0, :, :], gt[:, 0, :, :])  # compute loss
            loss_band2 = criterion(output3[:, 1, :, :], gt[:, 1, :, :])  # compute loss
            loss_band3 = criterion(output3[:, 2, :, :], gt[:, 2, :, :])  # compute loss
            loss_band4 = criterion(output3[:, 3, :, :], gt[:, 3, :, :])  # compute loss
            loss_band5 = criterion(output3[:, 4, :, :], gt[:, 4, :, :])  # compute loss
            loss_band6 = criterion(output3[:, 5, :, :], gt[:, 5, :, :])  # compute loss
            loss_band7 = criterion(output3[:, 6, :, :], gt[:, 6, :, :])  # compute loss
            loss_band8 = criterion(output3[:, 7, :, :], gt[:, 7, :, :])  # compute loss

            loss_band1_1 = criterion(output1[:, 0, :, :], gt[:, 0, :, :])  # compute loss
            loss_band2_1 = criterion(output1[:, 0, :, :], gt[:, 0, :, :])  # compute loss
            loss_band3_2 = criterion(output2[:, 2, :, :], gt[:, 2, :, :])  # compute loss
            loss_band4_2 = criterion(output2[:, 3, :, :], gt[:, 3, :, :])  # compute loss
            loss_band5_2 = criterion(output2[:, 4, :, :], gt[:, 4, :, :])  # compute loss
            loss_band6_2 = criterion(output2[:, 5, :, :], gt[:, 5, :, :])  # compute loss
            loss_band8_2 = criterion(output2[:, 6, :, :], gt[:, 7, :, :])  # compute loss



            loss_batch1 = criterion(output1, gt[:, [0,1], :, :])
            loss_batch2 = criterion(output2, gt[:, [0,1,2,3,4,5,7], :, :])###out2[0,1,2,3,4,5,7] gt[0,1,2,3,4,5,6,7]
            #loss_batch3 = criterion(output3, gt)
            loss_batch3 = loss_band1 + loss_band2 + loss_band3 + loss_band4 + loss_band5 + loss_band6 + loss_band7 + loss_band8

            # total loss:

            loss = loss_batch1 + loss_batch2 + loss_batch3


            epoch_train_loss.append(loss.item())

            # band-to-band loss:
            epoch_train_loss1.append(loss_band1.item())
            epoch_train_loss2.append(loss_band2.item())
            epoch_train_loss3.append(loss_band3.item())
            epoch_train_loss4.append(loss_band4.item())
            epoch_train_loss5.append(loss_band5.item())
            epoch_train_loss6.append(loss_band6.item())
            epoch_train_loss7.append(loss_band7.item())
            epoch_train_loss8.append(loss_band8.item())
            epoch_train_batch1.append(loss_batch1.item())
            epoch_train_batch2.append(loss_batch2.item())

            #experiment
            epoch_train_loss1_1.append(loss_band1_1.item())
            epoch_train_loss2_1.append(loss_band2_1.item())
            epoch_train_loss3_2.append(loss_band3_2.item())
            epoch_train_loss4_2.append(loss_band4_2.item())
            epoch_train_loss5_2.append(loss_band5_2.item())
            epoch_train_loss6_2.append(loss_band6_2.item())
            epoch_train_loss8_2.append(loss_band8_2.item())

            loss.backward()  # save all losses into a vector for one epoch
            optimizer.step()  # fixed

            ####print into txt

            # for name, layer in model.named_parameters():
            #     # writer.add_histogram('torch/'+name + '_grad_weight_decay', layer.grad, epoch*iteration)
            #     writer.add_histogram('net/' + name + '_data_weight_decay', layer, epoch * iteration)


        lr_scheduler.step()  # if update_lr, activate here!
        #print(optimizer.state_dict()['param_groups'][0]['lr'])

        t_loss_band1 = np.nanmean(np.array(epoch_train_loss1))  # compute the mean value of all losses, as one epoch loss
        writer.add_scalar('mse_loss/t_loss1', t_loss_band1, epoch)  # write to tensorboard to check

        t_loss_band2 = np.nanmean(np.array(epoch_train_loss2))
        writer.add_scalar('mse_loss/t_loss2', t_loss_band2, epoch)

        t_loss_band3 = np.nanmean(np.array(epoch_train_loss3))
        writer.add_scalar('mse_loss/t_loss3', t_loss_band3, epoch)

        t_loss_band4 = np.nanmean(np.array(epoch_train_loss4))
        writer.add_scalar('mse_loss/t_loss4', t_loss_band4, epoch)

        t_loss_band5 = np.nanmean(np.array(epoch_train_loss5))
        writer.add_scalar('mse_loss/t_loss5', t_loss_band5, epoch)

        t_loss_band6 = np.nanmean(np.array(epoch_train_loss6))
        writer.add_scalar('mse_loss/t_loss6', t_loss_band6, epoch)

        t_loss_band7 = np.nanmean(np.array(epoch_train_loss7))
        writer.add_scalar('mse_loss/t_loss7', t_loss_band7, epoch)

        t_loss_band8 = np.nanmean(np.array(epoch_train_loss8))
        writer.add_scalar('mse_loss/t_loss8', t_loss_band8, epoch)

        t_loss_batch1 = np.nanmean(np.array(epoch_train_loss8))
        t_loss_batch2 = np.nanmean(np.array(epoch_train_loss8))

        #experiment
        t_loss_band1_1 = np.nanmean(np.array(epoch_train_loss1_1))
        t_loss_band2_1 = np.nanmean(np.array(epoch_train_loss2_1))
        t_loss_band3_2 = np.nanmean(np.array(epoch_train_loss3_2))
        t_loss_band4_2 = np.nanmean(np.array(epoch_train_loss4_2))
        t_loss_band5_2 = np.nanmean(np.array(epoch_train_loss5_2))
        t_loss_band6_2 = np.nanmean(np.array(epoch_train_loss6_2))
        t_loss_band8_2 = np.nanmean(np.array(epoch_train_loss8_2))




        #########experiment
        file = open('separate_train_loss_band1_1.txt', 'a')  # write the training error into train_mse.txt
        file.write(str(t_loss_band1_1))
        file.write('\t')
        file.close()

        file = open('separate_train_loss_band2_1.txt', 'a')  # write the training error into train_mse.txt
        file.write(str(t_loss_band2_1))
        file.write('\t')
        file.close()

        file = open('separate_train_loss_band3_2.txt', 'a')  # write the training error into train_mse.txt
        file.write(str(t_loss_band3_2))
        file.write('\t')
        file.close()

        file = open('separate_train_loss_band4_2.txt', 'a')  # write the training error into train_mse.txt
        file.write(str(t_loss_band4_2))
        file.write('\t')
        file.close()

        file = open('separate_train_loss_band5_2.txt', 'a')  # write the training error into train_mse.txt
        file.write(str(t_loss_band5_2))
        file.write('\t')
        file.close()

        file = open('separate_train_loss_band6_2.txt', 'a')  # write the training error into train_mse.txt
        file.write(str(t_loss_band6_2))
        file.write('\t')
        file.close()

        file = open('separate_train_loss_band8_2.txt', 'a')  # write the training error into train_mse.txt
        file.write(str(t_loss_band8_2))
        file.write('\t')
        file.close()


        #########loss written
        file = open('separate_train_loss_band1.txt', 'a')  # write the training error into train_mse.txt
        file.write(str(t_loss_band1))
        file.write('\t')
        file.close()

        file = open('separate_train_loss_band2.txt', 'a')  # write the training error into train_mse.txt
        file.write(str(t_loss_band2))
        file.write('\t')
        file.close()

        file = open('separate_train_loss_band3.txt', 'a')  # write the training error into train_mse.txt
        file.write(str(t_loss_band3))
        file.write('\t')
        file.close()

        file = open('separate_train_loss_band4.txt', 'a')  # write the training error into train_mse.txt
        file.write(str(t_loss_band4))
        file.write('\t')
        file.close()

        file = open('separate_train_loss_band5.txt', 'a')  # write the training error into train_mse.txt
        file.write(str(t_loss_band5))
        file.write('\t')
        file.close()

        file = open('separate_train_loss_band6.txt', 'a')  # write the training error into train_mse.txt
        file.write(str(t_loss_band6))
        file.write('\t')
        file.close()

        file = open('separate_train_loss_band7.txt', 'a')  # write the training error into train_mse.txt
        file.write(str(t_loss_band7))
        file.write('\t')
        file.close()

        file = open('separate_train_loss_band8.txt', 'a')  # write the training error into train_mse.txt
        file.write(str(t_loss_band8))
        file.write('\t')
        file.close()

        file = open('separate_train_loss_batch1.txt', 'a')  # write the training error into train_mse.txt
        file.write(str(t_loss_batch1))
        file.write('\t')
        file.close()

        file = open('separate_train_loss_batch2.txt', 'a')  # write the training error into train_mse.txt
        file.write(str(t_loss_batch2))
        file.write('\t')
        file.close()

        # print total loss for each epoch
        #t_loss = np.nanmean(np.array(epoch_train_loss))
        print('Epoch: {}/{} training loss '.format(epochs, epoch))

        if epoch % ckpt == 0:  # if each ckpt epochs, then start to save model
            save_checkpoint(model, epoch)
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                output1, output2, output3 = model(test_ms, test_pan,test_lms)
                result_our = torch.squeeze(output3).permute(1, 2, 0)
                #sr = torch.squeeze(output3).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
                result_our = result_our * 2047
                result_our = result_our.type(torch.DoubleTensor).cuda()

                our_SAM, our_ERGAS = compute_index(test_gt, result_our, 4)
                print('our_SAM: {} dmdnet_SAM: 2.9355'.format(our_SAM) ) # print loss for each epoch
                print('our_ERGAS: {} dmdnet_ERGAS:1.8119 '.format(our_ERGAS))

        # ============Epoch Validate=============== #
        model.eval()
        with torch.no_grad():  # fixed
            for iteration, batch in enumerate(validate_data_loader, 1):
                gt, lms, ms, pan = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
                output1_v,output2_v,output3_v, = model(ms, pan,lms)
                loss_batch1_v = criterion(output1_v, gt[:, [0, 1], :, :])
                loss_batch2_v = criterion(output2_v, gt[:, [0, 1, 2, 3, 4, 5, 7], :, :])  ###out2[0,1,2,3,4,5,7] gt[0,1,2,3,4,5,6,7]
                loss_batch3_v = criterion(output3_v, gt)
                #loss_batch3 = loss_band1 + loss_band2 + loss_band3 + loss_band4 + loss_band5 + loss_band6 + loss_band7 + loss_band8
                epoch_val_batch1.append(loss_batch1_v.item())
                epoch_val_batch2.append(loss_batch2_v.item())
                epoch_val_batch3.append(loss_batch3_v.item())


        if epoch % 10 == 0:
            v_loss = []
            v_loss_batch1 = np.nanmean(np.array(epoch_val_batch1))  # compute the mean value of all losses, as one epoch loss
            writer.add_scalar('val/t_loss1', v_loss_batch1, epoch)  # write to tensorboard to check

            v_loss_batch2 = np.nanmean(np.array(epoch_val_batch2))
            writer.add_scalar('val/t_loss2', v_loss_batch2, epoch)

            v_loss_batch3 = np.nanmean(np.array(epoch_val_batch3))
            writer.add_scalar('val/t_loss3', v_loss_batch3, epoch)


            #print('validate loss: {}'.format(v_loss))
            file = open('separate_val_loss_batch1.txt', 'a')  # write the training error into train_mse.txt
            file.write(str(v_loss_batch1))
            file.write('\t')
            file.close()

            file = open('separate_val_loss_batch2.txt', 'a')  # write the training error into train_mse.txt
            file.write(str(v_loss_batch2))
            file.write('\t')
            file.close()

            file = open('separate_val_loss_batch3.txt', 'a')  # write the training error into train_mse.txt
            file.write(str(v_loss_batch3))
            file.write('\t')
            file.close()

    writer.close()  # close tensorboard


###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == "__main__":
    train_set = Dataset_Pro('/path/to/train/data')  # training data input
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    validate_set = Dataset_Pro('/path/to/validation/data')  # validation data input
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    # ------------------- load_test ----------------------------------#
    file_path = "/path/to/test/data" # test data input
    test_lms, test_ms, test_pan = load_set(file_path)
    test_lms = test_lms.cuda().unsqueeze(dim=0).float()
    test_ms = test_ms.cuda().unsqueeze(dim=0).float()  # convert to tensor type: 1xCxHxW (unsqueeze(dim=0))
    test_pan = test_pan.cuda().unsqueeze(dim=0).unsqueeze(dim=1).float()  # convert to tensor type: 1x1xHxW
    test_gt= load_gt_compared(file_path)  ##compared_result
    test_gt = (test_gt * 2047).cuda().double()
    ###################################################################

    train(training_data_loader, validate_data_loader,test_lms,test_ms, test_pan, test_gt)  # call train function (call: Line 66)


