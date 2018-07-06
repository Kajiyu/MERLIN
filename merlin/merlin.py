#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch as T
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from merlin.constants import *
from merlin.z_network import Z_network
from merlin.decoder import Decoder
from merlin.predictor import Predictor
from merlin.policy import Policy
from merlin.memory import Memory
from merlin.utils import *

class Merlin(nn.Module):
    def __init__(self):
        super(Merlin, self).__init__()
        self.z_network = Z_network()
        self.predictor = Predictor() #TODO: h outputs 2 layers [h1, h2]
        self.decoder = Decoder()
        self.policy = Policy()
        self.memory = Memory()

        self.mbp_loss_log = []
        self.policy_loss_log = []
    
    def reset(self, done=True):
        # Merlin holds variables shared between modules.
        self.rewards = []   # reward history
        self.log_pies = []  # action distribution (logarithmic) history
        self.V_preds = []   # value prediction history
        self.R_preds = []   # accumlated reward prediction history
        self.mbp_loss = 0       # MBP loss
        self.policy_loss = 0    # Policy loss

        if done:
            # episode finished
            self.predictor.reset()
            self.policy.reset()
            self.h = T.from_numpy(np.zeros((1,H_DIM), dtype=np.float32))
            self.m = T.from_numpy(np.zeros((1,M_DIM*Kr), dtype=np.float32))
            self.actions = [T.from_numpy(np.zeros(A_DIM, dtype=np.float32))]
        else:
            # episode goes on
            self.actions = [self.actions[-1]]

        self.action_indices = []  # for debug
    
    def step(self, o, r, time):
        # state variables
        o_, prev_a_, r_, = make_batch(o, self.actions[-1], r)    # add batch_size dimension
        z, p, q = self.z_network(o_, prev_a_, r_, self.h, self.m)

        # add KL divergence D(p||q) to loss
        self.mbp_loss += self._gaussian_kl_divergence(p, q)

        # policy process
        kp, bp = self.policy(z)
        # print("kp", kp.size())
        # print("bp", bp.size())
        mp = self.memory.read(kp, bp)
        log_pi, a = self.policy.get_action(z, mp)

        # MBP (predictor) process
        a_, = make_batch(a)
        self.h, k, b = self.predictor(z, a_, self.m)
        self.m = self.memory.read(k, b)

        # memory writing
        self.memory.write(z, time)

        # decoding
        o_dec, a_dec, r_dec, V_pred, R_pred = self.decoder(z, log_pi, a_)
        
        # add reconstruction error to loss
        self.mbp_loss += self._decode_loss(o_dec, o_, a_dec, prev_a_, r_dec, r_)

        # add history
        self.V_preds.append(V_pred.reshape(-1))
        self.R_preds.append(R_pred.reshape(-1))
        self.rewards.append(r)
        self.actions.append(a)
        self.log_pies.append(log_pi.reshape(-1))

        # return action index
        action = np.where(a==1)[0][0]
        self.action_indices.append(action)

        return action
    
    def _gaussian_kl_divergence(self, p, q):
        p_mean = p[0][:Z_DIM]
        p_logstd = p[0][Z_DIM:]
        p_var = T.sqrt(T.exp(p_logstd))
        q_mean = q[0][:Z_DIM]
        q_logstd = q[0][Z_DIM:]
        q_var = T.sqrt(T.exp(q_logstd))

        kl = (T.log(q_var/p_var) + (p_var + T.sqrt(p_mean-q_mean))/q_var - 1) * 0.5
        return T.sum(kl)
    
    def _decode_loss(self, o_dec, o, a_dec, a, r_dec, r):
        # WARNING: not taking mean in the paper.
        o_loss = F.mse_loss(o_dec, o)
        a_loss = self._bernoulli_softmax_crossentropy(a_dec, a)
        r_loss = F.mse_loss(r_dec, r) / 2
        return ALPHA_OBS*o_loss + ALPHA_ACTION*a_loss + ALPHA_REWARD*r_loss

    def _bernoulli_softmax_crossentropy(self, x, y):
        """ x: prediction. unnormalized distribution.
            y: teacher
        """
        return -T.sum(y * F.log_softmax(x, dim=-1) + (1-y) * T.log(1-F.softmax(x, dim=-1)+EPS))
    
    def update(self, done):
        print(self.action_indices)
        if done:
            # without bootstrap
            R_rev = [Variable(T.from_numpy(np.zeros(1, dtype=np.float32)))]
            A_rev = [Variable(T.from_numpy(np.zeros(1, dtype=np.float32)))]
            self.V_preds.append(0)
        else:
            # with bootstrap
            R_rev = [self.V_preds[-1]]
            A_rev = [Variable(T.from_numpy(np.zeros(1, dtype=np.float32)))]
            self.R_preds = self.R_preds[:-1]  # delete bootstrap element
            self.rewards = self.rewards[:-1]
        
        # accumulated rewards
        r_rev = self.rewards[::-1]
        for r in r_rev:
            R_rev.append(r + GAMMA*R_rev[-1])
        R = T.stack(R_rev[1:][::-1])   # (TRAIN_INTERVAL, 1)

        # advantages
        N = len(r_rev)
        assert len(self.V_preds) == N+1
        for i in range(N):
            delta = r_rev[i] + GAMMA*self.V_preds[N-i] - self.V_preds[N-i-1]
            A_rev.append(delta + GAMMA*LAMBDA*A_rev[-1])
        A = T.stack(A_rev[1:][::-1])

        # MBP loss
        V_preds = T.stack(self.V_preds[:-1])
        R_preds = T.stack(self.R_preds)
        #assert len(R) == len(R_preds) == len(V_preds) == len(A)
        R_loss = (T.sum(T.sqrt(V_preds - R)) + T.sum(T.sqrt(R_preds - R))) / 2
        self.mbp_loss += ALPHA_RETURN * R_loss
        self.mbp_loss *= ETA_MBP

        # Policy gradient
        A_ = 0
        H = 0
        self.actions = self.actions[1:]  # delete initial action
        for i in range(N):
            log_pi = self.log_pies[i]
            # log_pi*T.from_numpy(np.array(self.actions[i]==1).astype("float32"))
            # A_ += A[i] * log_pi[self.actions[i]==1]
            _t = log_pi*T.from_numpy(np.array(self.actions[i]==1).astype("float32"))
            A_ += A[i] * _t
            H += -T.matmul(T.exp(log_pi), log_pi)
        self.policy_loss -= A_[0] + ALPHA_ENTROPY*H     # gradient ascend
        self.policy_loss *= ETA_POLICY

        # update
        self.mbp_loss_log.append(self.mbp_loss.data)
        self.policy_loss_log.append(self.policy_loss.data)
    
    def backward(self):
        self.mbp_loss.backward()
        self.policy_loss.backward()
    
    def forward(self, done):
        self.update(done)