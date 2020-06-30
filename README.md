# SegGAN-Poem-torch-PARL
using PARL reinforement learning framework with torch to implement SegGAN(Chinese Poem generation)

original paper: [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/pdf/1609.05473v6.pdf)
## Introduction
这是一个用SeqGAN生成中文古诗的程序，使用百度的强化学习PARL框架以及pytorch。动态图如下（可以看到刚开始生成的序列结尾不太好，有很多“一”，后续逐渐变好）：

![](https://github.com/AddASecond/SegGAN-Poem-torch-PARL/blob/master/ReadMePic/poems.gif)

This is a project that using SeqGAN to generate Chinese Poem, where baidu's reinforcement learning framework PARL(with pytorch) are used.

github link of baidu's reinforcement learning framework PARL:
https://github.com/PaddlePaddle/PARL

## Arichitecture
This is how PARL abstracts RL as model-algorithm-agent:
![](https://github.com/AddASecond/SegGAN-Poem-torch-PARL/blob/master/ReadMePic/abstractions.png)

And this is how SeqGAN works:
![](https://github.com/AddASecond/SegGAN-Poem-torch-PARL/blob/master/ReadMePic/seqgan.png)

This is how I put SeqGAN into PARL framework:
* generator is actor/agent, generator.step gives "actions"(how to choose word), generator.sample (MTCS search in SeqGAN) gives the "states"(whole sequence samples) each episode(here one episode ends means the whole sequence are generated)

* discriminator and rollout are critic/environments, which obtain samples/embedding, output rewards  

* rewards(loss) are used to train critic/env(discriminator) and actor/agent(generator)

* All PARL-related codes are used in train_generator_PG in main function

## TODOs
* train using Poems as corpus   -Done
* using PARL framework   -Done
* using build-in functions in PARL to substitude some function   -ing

## Thanks
most of code borrow from https://github.com/X-czh/SeqGAN-PyTorch and https://github.com/TobiasLee/SeqGAN_Poem, but merge them into PARL framework for better 
understanding of the RL process in SeqGAN.
