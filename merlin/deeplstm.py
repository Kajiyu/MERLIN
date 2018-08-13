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

class DeepLSTM(nn.Module):
    def __init__(self, d_in, d_out):
        super(DeepLSTM, self).__init__()
        self.l1 = nn.LSTM(d_in, d_out)
        self.l2 = nn.Linear(d_out, d_out)
        self.hidden_dim = d_out
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        return (T.zeros(1, 1, self.hidden_dim), T.zeros(1, 1, self.hidden_dim))
    
    def reset_state(self):
        self.hidden = self.init_hidden()
    
    def forward(self, x):
        self.x = x
        x, self.hidden = self.l1(self.x, self.hidden)
        self.y = self.l2(x)
        return self.y