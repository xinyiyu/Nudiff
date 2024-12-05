# Nudiff: Diffusion-based Data Augmentation for Nuclei Image Segmentation
This repository holds the code for the paper "Diffusion-based data augmentation for nuclei image segmentation".

## Description
Nuclei segmentation is a fundamental but challenging task in the quantitative analysis of histopathology images. Although fullysupervised deep learning-based methods have made significant progress, a large number of labeled images are required to achieve great segmentation performance. Considering that manually labeling all nuclei instances for a dataset is inefficient, obtaining a large-scale human-annotated dataset is time-consuming and labor-intensive. Therefore, augmenting a dataset with only a few labeled images to improve the segmentation performance is of significant research and application value. In this paper, we introduce the first diffusion-based augmentation method for nuclei segmentation. The idea is to synthesize a large  number of labeled images to facilitate training the segmentation model. To achieve this, we propose a two-step strategy. In the first step, we train an unconditional diffusion model to synthesize the Nuclei Structure that is defined as the representation of pixel-level semantic and distance transform. Each synthetic nuclei structure will serve as a constraint on histopathology image synthesis and is further post-processed to be an instance map. In the second step, we train a conditioned diffusion model to synthesize histopathology images based on nuclei structures. The synthetic histopathology images paired with synthetic instance maps will be added to the real dataset for training the segmentation model. The experimental results show that by augmenting 10% labeled real dataset with synthetic samples, one can achieve comparable segmentation results with the fully-supervised baseline.

## Installation
Clone this repository and install the package in your terminal under the `Nudiff` directory:
```
python setup.py develop
```
The `./hover_net` directory holds the code of [vqdang/hover-net](https://github.com/vqdang/hover_net) with slight modifications. Please also read the `README.md` in `./hover_net`.

## Data
The experiments in our paper are conducted on two datasets: MoNuSeg and Kumar. The MoNuSeg dataset has 44 labeled images of size 1000 × 1000, 30 for training and 14 for testing. The Kumar dataset consists of 30 1000 ×1000 labeled images from seven organs of The Cancer Genome Atlas (TCGA) database. The dataset is splited into 16 training images and 14 testing images. Download the two datasets from [link](https://drive.google.com/drive/folders/1l1gb2gu8nJL7LEITjHCN0NQeBNXFGAxH?usp=sharing).

## Usage
The [notebook](https://github.com/xinyiyu/Nudiff/blob/main/nudiff_note.ipynb) shows the pipeline of data preparation, diffusion models training, sampling and segmentation model training/testing. Please read it before you use the package.

## Pretrained models
The pretrained unconditional and conditional diffusion models of MoNuSeg and Kumar can be downloaded from [baiduyun](https://pan.baidu.com/s/1pwTfYQ_lvly32Mi8Xo1KHg?pwd=isjg).

## Acknowledge
The repository is mainly based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion) and [vqdang/hover-net](https://github.com/vqdang/hover_net). We also thank for the contributors of public datasets.

## Citation
Please consider citing our paper if you find it helpful to your research:
```
@inproceedings{yu2023diffusion,
  title={Diffusion-Based Data Augmentation for Nuclei Image Segmentation},
  author={Yu, Xinyi and Li, Guanbin and Lou, Wei and Liu, Siqi and Wan, Xiang and Chen, Yan and Li, Haofeng},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={592--602},
  year={2023},
  organization={Springer}
}
```

## Contact information
Please contact Xinyi Yu (xyyu98@gmail.com) and Prof. Haofeng Li (lihaofeng@cuhk.edu.cn) if any enquiry.

