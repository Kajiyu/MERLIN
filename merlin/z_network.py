#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch as T
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from merlin.constants import *

class Z_network(nn.Module):
    def __init__(self):
        super(Z_network, self).__init__()
        self.o_encoder1 = nn.Linear(O_DIM, 12)
        self.o_encoder2 = nn.Linear(12, 6)
        self.prior1 = nn.Linear(H_DIM+M_DIM*Kr, 2*Z_DIM)
        self.prior2 = nn.Linear(2*Z_DIM, 2*Z_DIM)
        self.prior3 = nn.Linear(2*Z_DIM, 2*Z_DIM)
        self.f_post1 = nn.Linear(6+A_DIM+1+H_DIM+M_DIM*Kr+2*Z_DIM, 2*Z_DIM)
        self.f_post2 = nn.Linear(2*Z_DIM, 2*Z_DIM)
        self.f_post3 = nn.Linear(2*Z_DIM, 2*Z_DIM)
    
    def forward(self, o, a, r, h, m):
        o_encode = nn.ReLU(True)(self.o_encoder1(o))
        o_encode = nn.ReLU(True)(self.o_encoder2(o_encode))
        e = T.cat([o_encode, a, r], 1)
        state = T.cat([h, m], 1)
        prior = nn.Tanh()(self.prior1(state))
        prior = nn.Tanh()(self.prior2(prior))
        prior = self.prior3(prior)

        n = T.cat([e, h, m, prior], 1)
        f_post = nn.Tanh()(self.f_post1(n))
        f_post = nn.Tanh()(self.f_post2(f_post))
        f_post = self.f_post3(f_post)
        posterior = prior + f_post

        gaussian = T.from_numpy(np.random.normal(size=(1,Z_DIM)).astype(np.float32))
        z = posterior[:, :Z_DIM] + T.exp(posterior[:, Z_DIM:]) * gaussian
        return z, prior, posterior
