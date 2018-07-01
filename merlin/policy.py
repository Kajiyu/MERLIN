#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch as T
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from merlin.constants import *
from merlin.deeplstm import DeepLSTM

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.policy = DeepLSTM(Z_DIM+M_DIM*Krp, Hp_DIM)
        self.reader = nn.Linear(Hp_DIM, Krp*(M_DIM+1))
        self.pi1 = nn.Linear(Z_DIM+Hp_DIM+M_DIM*Krp, 200)
        self.pi2 = nn.Linear(200, A_DIM)
        self.soft_plus = nn.Softplus()
    
    def reset(self):
        self.policy.reset_state()
        self.h = np.zeros((1, Hp_DIM), dtype=np.float32)
        self.m = np.zeros((1, M_DIM*Krp), dtype=np.float32)
    
    def forward(self, z):
        state = T.cat([z, self.m], 1)
        self.h = self.policy(state)
        i = self.reader(self.h)
        k = i[:, :M_DIM*Krp]
        sc = i[:, M_DIM*Krp:]
        b = self.soft_plus(sc, 1)
        return k, b
    
    def get_action(self, z, m):
        # assert m.shape == (1, M_DIM * Krp)
        self.m = m
        state = T.cat([z.data, self.h, m], 1)   # Stop gradients wrt z.
        state = nn.Tanh()(self.pi1(state))
        log_pi = nn.LogSoftmax()(self.pi2(state)) # log_softmax may be more stable.
        probs = T.exp(log_pi)[0]
        
        # avoid "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
        diff = sum(probs.data[:-1]) - 1
        if diff > 0:
            probs -= (diff + np.finfo(np.float32).epsneg) / (A_DIM - 1)

        a = np.random.multinomial(1, probs.data).astype(np.float32) # onehot
        return log_pi, a