## DeepLabV3+ Pytorch Lightning Implementation

### Introduction
This repo ports YudeWang's excellent [repo](https://github.com/YudeWang/deeplabv3plus-pytorch) for deeplabv3plus in Pytorch to Pytorch-Lightning. 
We reproduce performance near the reported performance on Pascal VOC 2012 and Cityscapes.

The major changes are: 
1) Remove the usage of SyncBatchNorm, pytorch lightning handles this now
2) Adding logging with the WandB logger
3) Added support to run on a single GPU

### Usage
Run setup.sh to build a virtual environment and install the required packages. By default, this will also 
download the Pascal Dataset. If you'd like to download cityscapes uncomment the appropriate lines in setup.sh.

Then execute the run_pascal.sh script with the desired number of gpus, batch size and epochs. 

### Results

Our Cityscapes result is for the Resnet-101 backbone with atrous convolutions, unfortunately in the paper we only have
the result for the Xception backbone.

| Dataset |      Backbone      | val mIoU | val mIoU (paper) |
| :---: |:------------------:|:--------:|:----------------:|
| Pascal VOC 2012 |     ResNet-101     |  78.57   |      78.85       |
| Cityscapes |     ResNet-101     |   76.8   |      79.55*      |

