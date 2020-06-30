#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#-*- coding: utf-8 -*-

import paddle.fluid as fluid
import parl
from parl import layers
from loss import PGLoss
import torch.optim as optim


class PolicyGradient(parl.Algorithm):
    def __init__(self, model, lr=None):
        """ Policy Gradient algorithm

        Args:
            model (parl.Model): policy的前向网络.
            lr (float): 学习率.
        """

        self.model = model
        assert isinstance(lr, float)
        self.lr = lr

    def predict(self, obs):
        """ 使用policy model预测输出的动作概率
        """
        return self.model.generator(obs)

    def learn(self, obs, action, reward):
        """ 用policy gradient 算法更新policy model
        """

        # update generator
        act_prob = self.predict(obs)  # 获取generator输出的选词动作概率
        loss = PGLoss(act_prob, action, reward)
        gen_optimizer = optim.Adam(params=self.model.generator.parameters(), lr=self.lr)
        gen_optimizer.zero_grad()
        loss.backward()
        gen_optimizer.step()