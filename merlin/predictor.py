#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch as T
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from merlin.constants import *
from merlin.deeplstm import DeepLSTM

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.predictor = DeepLSTM(Z_DIM+A_DIM+M_DIM*Kr, H_DIM)
        self.reader = nn.Linear(H_DIM, Kr*(M_DIM+1))
        self.soft_plus = nn.Softplus()
    
    def reset(self):
        self.predictor.reset_state()
    
    def forward(self, z, a, m):
        state = T.cat([z, a, m], 1).view(-1, 1, Z_DIM+A_DIM+M_DIM*Kr)
        h = self.predictor(state).view(-1, H_DIM)
        i = self.reader(h)
        k = i[:, :M_DIM*Kr]
        sc = i[:, M_DIM*Kr:]
        b = self.soft_plus(sc)
        return h, k, b