# GCLNet
The code in this toolbox implements the "Graph-based context learning network for infrared small target detection." More detailed information is provided below.

## Requirements
* Python 3.7
* pytorch 1.11.0

## Datasets
* NUAA-SIRST [[download]](https://github.com/YimianDai/sirst)
* NUDT-SIRST [[download]](https://github.com/YeRen123455/Infrared-Small-Target-Detection)

## Pretrained Weight  
The pretrained weights need to be placed in the corresponding folder `./log/NUAA-SIRST/GCLNet_best.pth.tar` and `./log/NUDT-SIRST/GCLNet_best.pth.tar`.  
[Google Drive](https://drive.google.com/drive/folders/1U9y5lHmdOv5NFnhCnI36uBHRniPeaLP1?usp=sharing)  
[Baidu Drive](https://pan.baidu.com/s/1P10lvyztFD6r5k5k7iOBvw) (access code: hh16)  


## Experiments
The training and testing experiments are conducted using PyTorch with a single GeForce RTX 3080 GPU.
