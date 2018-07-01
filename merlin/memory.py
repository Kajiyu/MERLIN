#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch as T
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from merlin.constants import *

class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()
        self.size = 1
        self.M = Variable(np.zeros((N_mem, M_DIM), dtype=np.float32))
        self.W_predictor = None
        self.W_policy = None
        self.v_wr = Variable(np.zeros((N_mem, 1), dtype=np.float32))
        self.v_ret = Variable(np.zeros((N_mem, 1), dtype=np.float32))
        self.u = Variable(np.zeros((1, N_mem), dtype=np.float32))
    
    def read(self, k, b):
        # k: (1, M_DIM*kr), b: (1, kr)
        kr = b.size(1)
        K = k.view(kr, M_DIM)
        _K = K + EPS # in chainer, F.normalize(K + EPS)
        _M = self.M + EPS # in chainer, F.normalize(self.M + EPS)
        C = T.matmul(_K, T.transpose(_M))
        B = F.repeat(b, N_mem).view(kr, N_mem)  # beta
        if kr == Kr:
            self.W_predictor = F.softmax(B*C)  # B*C: elementwise multiplication
            M = T.matmul(self.W_predictor, self.M)
        elif kr == Krp:
            self.W_policy = F.softmax(B*C)
            M = T.matmul(self.W_policy, self.M)
        else:
            raise(ValueError)
        return M.view(1, -1)