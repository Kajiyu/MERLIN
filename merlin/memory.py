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
        self.M = Variable(T.from_numpy(np.zeros((N_mem, M_DIM), dtype=np.float32)))
        self.W_predictor = None
        self.W_policy = None
        self.v_wr = Variable(T.from_numpy(np.zeros((N_mem, 1), dtype=np.float32)))
        self.v_ret = Variable(T.from_numpy(np.zeros((N_mem, 1), dtype=np.float32)))
        self.u = Variable(T.from_numpy(np.zeros((1, N_mem), dtype=np.float32)))
    
    def read(self, k, b):
        # k: (1, M_DIM*kr), b: (1, kr)
        kr = b.size(1)
        K = k.view(kr, M_DIM)
        _K = F.normalize(K, eps=EPS)
        _M = F.normalize(self.M, eps=EPS)
        C = T.matmul(_K, T.transpose(_M, 0, 1))
        B = b.repeat(N_mem, 1)  # beta
        B = T.transpose(B, 0, 1)
        if kr == Kr:
            self.W_predictor = F.softmax(B*C, dim=1)  # B*C: elementwise multiplication
            M = T.matmul(self.W_predictor, self.M)
        elif kr == Krp:
            self.W_policy = F.softmax(B*C, dim=1)
            M = T.matmul(self.W_policy, self.M)
        else:
            raise(ValueError)
        return M.view(1, -1)
    
    def write(self, z, time, debug=False):
        # update usage indicator
        self.u += T.matmul(Variable(T.from_numpy(np.ones((1, Kr), dtype=np.float32))), self.W_predictor)

        # update writing weights
        prev_v_wr = self.v_wr
        v_wr = np.zeros((N_mem, 1), dtype=np.float32)
        if time < N_mem:
            v_wr[time][0] = 1
        else:
            waste_index = int(T.argmin(self.u).data)
            v_wr[waste_index][0] = 1
        self.v_wr = Variable(T.from_numpy(v_wr))

        # writing
        # z: (1, Z_DIM)
        if debug:
            print(self.M)
        if USE_RETROACTIVE:
            # update retroactive weights
            self.v_ret = GAMMA*self.v_ret + (1-GAMMA)*prev_v_wr
            z_wr = T.cat([z, Variable(T.from_numpy(np.zeros((1, Z_DIM), dtype=np.float32)))], 1)
            z_ret = T.cat([Variable(T.from_numpy(np.zeros((1, Z_DIM), dtype=np.float32))), z], 1)
            self.M += T.matmul(self.v_wr, z_wr) + T.matmul(self.v_ret, z_ret)
        else:
            self.M += T.matmul(self.v_wr, z)
        if debug:
            return self.M