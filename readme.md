# **PROGRESSIVE BAND-SEPARATED CONVOLUTIONAL NEURAL NETWORK FOR MULTISPECTRAL PANSHARPENING**

This repository contains main code for the paper *PROGRESSIVE BAND-SEPARATED CONVOLUTIONAL NEURAL NETWORK FOR MULTISPECTRAL PANSHARPENING,* contributed by Shishi Xiao, Cheng Jin, Tian-Jing Zhang, Ran Ran, and Liang-Jian Deng. All rights reserved by authors.

Homepage: 

[https://serendipitysx.github.io/](https://serendipitysx.github.io/)

[https://chengjin-git.github.io/](https://chengjin-git.github.io/)

[https://liangjiandeng.github.io/](https://liangjiandeng.github.io/)

## Introduction

 In this paper, we design a progressive, band-separated convolutional network architecture for discriminatively learning the features and relation among spectral bands, aiming to address the problem mentioned before. More specififically, the proposed architecture mainly consists of three aspects. First, to accurately preserve the spectral peculiarities, we divide the multispectral input image in terms of its bands into several groups. Second, our original panchromatic and multispectral inputs are fifiltered by a high-pass operation to further yield more spatial details. Third, we use a spectral fusion module (SFM) for each group and associate them to progressively assemble the whole architecture. It is worth mentioning that the architecture could be integrated into any other competitive CNNs to improve the performance. 

## Dependencies and Installation
- Python 3.8 (Recommend to use Anaconda)
- PyTorch > 1.1
- NVIDIA GPU + CUDA
- Python packages: `pip install numpy scipy h5py`
- TensorBoard

## Dataset Preparation
The datasets used in this paper is WorldView-3 (can be downloaded [here](https://www.maxar.com/product-samples/)), QuickBird (can be downloaded [here](https://earth.esa.int/eogateway/catalog/quickbird-full-archive)) and GaoFen-2 (can be downloaded [here](http://www.rscloudmart.com/dataProduct/sample)). Due to the copyright of dataset, we can not upload the datasets, you may download the data and simulate them according to the paper.

## PBSNet Architecture


## Results

## Citation
```bibtex
@INPROCEEDINGS{psbnnet,
  author={Xiao, Shi-Shi and Jin, Cheng and Zhang, Tian-Jing and Ran, Ran and Deng, Liang-Jian},
  booktitle={2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS}, 
  title={Progressive Band-Separated Convolutional Neural Network for Multispectral Pansharpening}, 
  year={2021},
  volume={},
  number={},
  pages={4464-4467},
  doi={10.1109/IGARSS47720.2021.9554024}}
```

## Contact
We are glad to hear from you. If you have any questions, please feel free to contact <xrakexss@gmail.com> or open issues on this repository.
