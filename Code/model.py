# This is a pytorch version for the work of PanNet
# YW Jin, X Wu, LJ Deng(UESTC);
# 2020-09;

import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as int


# -------------Initialization----------------------------------------
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):  ## initialization for Conv2d
                # try:
                #     import tensorflow as tf
                #     tensor = tf.get_variable(shape=m.weight.shape, initializer=tf.variance_scaling_initializer(seed=1))
                #     m.weight.data = tensor.eval()
                # except:
                #     print("try error, run variance_scaling_initializer")
                # variance_scaling_initializer(m.weight)
                variance_scaling_initializer(m.weight)  # method 1: initialization
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # method 2: initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):  ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):  ## initialization for nn.Linear
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # 如果mode = "fan_in"， n为输入单元的结点数
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


# -------------ResNet Block (One)----------------------------------------
class Resblock(nn.Module):
    def __init__(self):
        super(Resblock, self).__init__()

        channel = 32
        self.conv20 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=5, stride=1, padding=2,
                                bias=True)
        self.conv21 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=5, stride=1, padding=2,
                                bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.relu(self.conv20(x))  # Bsx32x64x64
        rs1 = self.conv21(rs1)  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs


# -----------------------------------------------------
class OURNet(nn.Module):
    def __init__(self):
        super(OURNet, self).__init__()

        channel = 32
        spectral_num = 8

        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize #outpading=stride-1 #（16-1）*4-4+8=64
        self.deconv = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=8, stride=4,
                                         padding=2, bias=True)

        self.conv1 = nn.Conv2d(in_channels=9, out_channels=32, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv4 = nn.Conv2d(in_channels=9, out_channels=32, kernel_size=3, stride=1, padding=1,
                              bias=True)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=7, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv6 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv7 = nn.Conv2d(in_channels=9, out_channels=32, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv8 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=1, padding=1,
                               bias=True)

        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()
        self.res5 = Resblock()
        self.res6 = Resblock()
        self.res7 = Resblock()
        self.res8 = Resblock()
        self.res9 = Resblock()
        self.res10 = Resblock()
        self.res11= Resblock()
        self.res12 = Resblock()

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(32)

        # backbone = []     # method 1: 4 resnet repeated blocks
        # for i in range(4):
        #     backbone.append(self.res)
        # self.backbone = nn.Sequential(*backbone)


        self.backbone1 = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            self.res2,
            self.res3,
            self.res4,
        )
        self.backbone2 = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res5,
            self.res6,
            self.res7,
            self.res8,
        )
        self.backbone3 = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res9,
            self.res10,
            self.res11,
            self.res12,

        )


        #init_weights(self.backbone1,self.backbone2,self.backbone3, self.deconv,\
         #             self.conv3,self.conv6)  # state initialization, important!

    def forward(self, x, y, lms):  # x=  ms; y = pan
        ##data dealing
        ms = x
        ##lms
        lms_split = torch.split(lms, [2, 4, 1, 1], dim=1)  # lms1, lms2, lms3, lms4, lms5, lms6, lms7, ms8
        lms_batch1 = lms_split[0]
        lms_batch2 = torch.cat((lms_split[1], lms_split[3]), 1)
        
        ##net1
        output_deconv = self.deconv(ms)
        input = torch.cat([output_deconv, y], 1)  # Bsx9x64x64
        rs = self.conv1(input)
        rs = self.relu(rs)      # Bsx32x64x64##########  9/32
        rs = self.backbone1(rs)  # output is Bsx32x64x64
        rs = self.conv2(rs) # Bsx2x64x64 32/2
        output1 = rs + lms_batch1  # Bsx2x64x64 32/2

        ##net2
        input2 = torch.cat((output1, output_deconv[:, [2, 3, 4, 5, 6, 7], :, :]), 1)  # Bsx(2+6)x64x64
        input2 = self.conv3(input2) # Bsx(8)x64x64  8/8
        input2 = torch.cat((input2, y), 1)  # Bsx(8+1)x64x64
        rs = self.conv4(input2)
        rs = self.relu(rs)  # Bsx32x64x64 9/32
        rs = self.backbone2(rs)  # output is Bsx32x64x64
        rs = self.conv5(rs) # Bsx2x64x64 32/7
        output2 = rs + torch.cat((lms_batch1,lms_batch2),dim=1)  # Bsx7x64x64 32/7

        ##net3
        input3 = torch.cat((output2, output_deconv[:, 6, :, :].unsqueeze(1)), 1)  # Bsx(7+1)x64x64
        input3 = self.conv6(input3)  # Bsx(8)x64x64  8/8
        input3 = torch.cat((input3, y), 1)  # Bsx(9)x64x64
        rs = self.conv7(input3)
        rs = self.relu(rs)  # Bsx32x64x64  9/32
        rs = self.backbone3(rs)   # output is Bsx32x64x64
        output3 = self.conv8(rs) + lms  # Bsx8x64x64  32/8
        return output1, output2, output3


# ----------------- End-Main-Part ------------------------------------
def variance_scaling_initializer(tensor):
    from scipy.stats import truncnorm

    def truncated_normal_(tensor, mean=0, std=1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def variance_scaling(x, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(x)
        if mode == "fan_in":
            scale /= max(1., fan_in)
        elif mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        if distribution == "normal" or distribution == "truncated_normal":
            # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = math.sqrt(scale) / .87962566103423978
        # print(fan_in,fan_out,scale,stddev)#100,100,0.01,0.1136
        truncated_normal_(x, 0.0, stddev)
        return x / 10 * 1.28

    variance_scaling(tensor)

    return tensor


def summaries(model, writer=None, grad=False):
    if grad:
        from torchsummary import summary
        summary(model, input_size=[(8, 16, 16), (1, 64, 64), (8, 64, 64)], batch_size=1)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

    if writer is not None:
        x = torch.randn(1, 64, 64, 64)
        writer.add_graph(model, (x,))


def inspect_weight_decay():
    ...
