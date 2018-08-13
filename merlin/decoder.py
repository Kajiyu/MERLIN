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

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder1 = nn.Linear(Z_DIM, 2*Z_DIM)
        self.decoder2 = nn.Linear(2*Z_DIM, 2*Z_DIM)
        self.decoder3 = nn.Linear(2*Z_DIM, 6+A_DIM+1+H_DIM+M_DIM+2*Z_DIM)
        self.decoder4 = nn.Linear(6+A_DIM+1+H_DIM+M_DIM+2*Z_DIM, 6+A_DIM+1)
        self.o_decoder1 = nn.Linear(6, 12)
        self.o_decoder2 = nn.Linear(12, O_DIM)
        self.a_decoder = nn.Linear(A_DIM, A_DIM)
        
        self.value1 = nn.Linear(Z_DIM+A_DIM, 200)
        self.value2 = nn.Linear(200, 1)
        self.advantage1 = nn.Linear(Z_DIM+A_DIM, 50)
        self.advantage2 = nn.Linear(50, 1)
    
    def forward(self, z, log_pi, a):
        decode = nn.ReLU()(self.decoder1(z))
        decode = nn.ReLU()(self.decoder2(decode))
        decode = nn.ReLU()(self.decoder3(decode))
        decode = self.decoder4(decode)
        o_decode = nn.ReLU()(self.o_decoder1(decode[:, :6]))
        o_decode = self.o_decoder2(o_decode)
        a_decode = self.a_decoder(decode[:, 6:6+A_DIM]) # softmax or onehoten? →loss計算時にやる
        r_decode = decode[:, -1:]

        state = T.cat([z, log_pi], 1)
        V = nn.Tanh()(self.value1(state))
        V = self.value2(V)

        state = T.cat([z, a], 1)
        A = nn.Tanh()(self.advantage1(state))
        A = nn.Tanh()(self.advantage2(A))

        R = V.data + A  # stop gradient wrt V
        return o_decode, a_decode, r_decode, V, R

