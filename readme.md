# **PROGRESSIVE BAND-SEPARATED CONVOLUTIONAL NEURAL NETWORK FOR MULTISPECTRAL PANSHARPENING**

This repository contains main code for the paper *PROGRESSIVE BAND-SEPARATED CONVOLUTIONAL NEURAL NETWORK FOR MULTISPECTRAL PANSHARPENING,* contributed by Shishi Xiao, Cheng Jin, Tianjing Zhang, Ran Ran, and Liangjian Deng. All rights reserved by authors.

## Introduction

 In this paper, we design a progressive, band-separated convolutional network architecture for discriminatively learning the features and relation among spectral bands, aiming to address the problem mentioned before. More specififically, the proposed architecture mainly consists of three aspects. First, to accurately preserve the spectral peculiarities, we divide the multispectral input image in terms of its bands into several groups. Second, our original panchromatic and multispectral inputs are fifiltered by a high-pass operation to further yield more spatial details. Third, we use a spectral fusion module (SFM) for each group and associate them to progressively assemble the whole architecture. It is worth mentioning that the architecture could be integrated into any other competitive CNNs to improve the performance. 