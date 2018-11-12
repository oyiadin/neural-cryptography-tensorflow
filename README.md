# HCTF 2018 - [Misc] Guess My Key

比赛过程中使用的模型是预先训练好的，本仓库包含了训练所需代码以及 exp。writeup 之后再更新。

The model used during the competition was trained in advance. This repo contains all the codes needed for training, and also my exp. I am going to update my writeup here in a few days.


# Adversarial Neural Cryptography in [TensorFlow](https://github.com/tensorflow/tensorflow)

A Tensorflow Flow implementation of Google Brain's recent paper ([Learning to Protect Communications with Adversarial Neural Cryptography.](https://arxiv.org/pdf/1610.06918v1.pdf))

Two Neural Networks, Alice and Bob learn to communicate secretly with each other, in presence of an adversary Eve.

![Setup](assets/diagram.png)

## Pre-requisites

* TensorFlow 
* Numpy

## Usage 
First, ensure you have the dependencies installed.

    $ pip install -r requirements.txt

To train the neural networks, run the `main.py` script.

    $ python main.py --msg-len 96 --epochs 60
    
    
## Attribution / Thanks

* carpedm20's DCGAN [implementation](https://github.com/carpedm20/DCGAN-tensorflow) in TensorFlow. 
* Liam's [implementation](https://github.com/nlml/adversarial-neural-crypt) of Adversarial Neural Cryptography in Theano. 

## Citing Code
If you want to cite this code in your work, refer to the following DOI:

[![DOI](https://zenodo.org/badge/73807045.svg)](https://zenodo.org/badge/latestdoi/73807045)

## License

MIT
