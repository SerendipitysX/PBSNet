# PROGRESSIVE BAND-SEPARATED CONVOLUTIONAL NEURAL NETWORK FOR MULTISPECTRAL PANSHARPENING

Homepage: 

[https://serendipitysx.github.io/](https://serendipitysx.github.io/)

[https://chengjin-git.github.io/](https://chengjin-git.github.io/)

[https://liangjiandeng.github.io/](https://liangjiandeng.github.io/)

- Code for paper: "PROGRESSIVE BAND-SEPARATED CONVOLUTIONAL NEURAL NETWORK FOR MULTISPECTRAL PANSHARPENING"
- State-of-the-art pansharpening performance

This is the description of how to run our training code and testing code. 

## Training instructions

In `train.py`:

- Modify the train/validation data input in .mat format
- Modify the data path in line `383`, `387`.
- Change the normalizaition scale according to the worldview level of the corresponding datasets

In `model.py`:
- Modify `num_feature`, `num_ms_channels`
- Change output channel number of image alignment step

## Testing instructions

In `model.py`:
- Modify `num_feature`, `num_ms_channels` according to the dataset settings in the comments
- Modify the data path in line `392`.
- Change output channel number of image alignment step according to the dataset settings in the comments


In `test_single.py`:
- Load (pretrained) model path
- Load test data
- Change the MTF kernel input according to the dataset settings in the comments

## Third-Party datasets stucture
PBSNet supports datasets other than WorldView-3, QuickBird and GaoFen-2. If you want to test on your own dataset, make sure to convert your dataset into `.mat` format and contains the following sturcture:

```
YourDataset.mat
|--ms: original multispectral images in .mat format, basically have the size of N*C*h*w 
|--lms: interpolated multispectral images in .mat format, basically have the size of N*C*H*W 
|--pan: original panchromatic images in .mat format, basically have the size of N*1*H*W 
|--gt: simulated ground truth images images in .mat format, basically have the size of N*C*H*W 
```

where `H=nh, W=nw`, n denotes the upsampling factor.
