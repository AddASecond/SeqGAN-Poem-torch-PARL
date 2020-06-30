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

import numpy as np
import paddle.fluid as fluid
import parl
from parl import layers
import torch


class Agent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        self.batch_size = obs_dim
        self.g_seq_len = act_dim
        super(Agent, self).__init__(algorithm)

    def build_program(self):
        pass

    def sample(self):
        samples = self.algorithm.model.generator.sample(self.batch_size, self.g_seq_len)
        zeros = torch.zeros(self.batch_size, 1, dtype=torch.int64)
        if samples.is_cuda:
            zeros = zeros.cuda()
        inputs = torch.cat([zeros, samples.data], dim = 1)[:, :-1].contiguous() # here produce generator inputs
        targets = samples.data.contiguous().view((-1,))
        self.obs = inputs
        self.act = targets
        return samples


    def predict(self):
        self.algorithm.predict(self.obs)

    def learn(self, reward, optimizer):
        self.algorithm.learn(self.obs, self.act, reward)