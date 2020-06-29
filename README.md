# SegGAN-Poem-torch-PARL
using PARL reinforement learning framework with torch to implement SegGAN(Chinese Poem generation)
## Introduction
这是一个用SeqGAN生成中文古诗的程序，使用百度的强化学习PARL框架以及pytorch。

This is a project that using SeqGAN to generate Chinese Poem, where baidu's reinforcement learning framework PARL(with pytorch) are used.

github link of baidu's reinforcement learning framework PARL:
https://github.com/PaddlePaddle/PARL

![This is how PARL abstracts RL as model-algorithm-agent:](https://github.com/AddASecond/SegGAN-Poem-torch-PARL/blob/master/ReadMePic/abstractions.png)

![And this is how SeqGAN works:](https://github.com/AddASecond/SegGAN-Poem-torch-PARL/blob/master/ReadMePic/seqgan.png)

This is how I put SeqGAN into PARL framework:


## Thanks
most of code borrow from https://github.com/X-czh/SeqGAN-PyTorch and https://github.com/TobiasLee/SeqGAN_Poem, but merge them into PARL framework for better 
understanding of the RL process in SeqGAN.
