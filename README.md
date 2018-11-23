# Image-to-Image Translation via Supervised Conditional GAN

## Framework

![Framework](https://github.com/truongthanhdat/CondGAN/raw/master/figures/framework.png)

The method is based on GAN. The generative network uses Variational Autoencoders (VAE) which includes a stack of convolutional layers, deconvolutional and residual layers. The disrciminative network includes a stack of convolutional layers. The network use VGG-16 trained on Imagenet to compute perceptual loss between predicted images and groundtruth images.

## Requirements

+ python 2.7
+ tensorflow
+ easydict
```bash
pip install -r requirements.txt
```

## Usage

For more detail, you can see in the source code.

To train the network, please use train.py.  For more detail, type:
```bash
python train.py --help
```
To test the network , pleasee use test.py.  For more detail, type:
```bash
python test.py --help
```

## Demo

Demo of Gray Image to Color Image uses Image Transaltion via Supervised Conditional GAN



[![Demo](https://github.com/truongthanhdat/CondGAN/raw/master/figures/title.png)](https://youtu.be/ki45TWttFVE)


## Author
Thanh-Dat Truong,
University of Science, VNU-HCM
Email: ttdat@selab.hcmus.edu.vn
