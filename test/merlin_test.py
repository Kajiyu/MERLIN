#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch as T
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from merlin.constants import *
from merlin.merlin import Merlin

def main():
    T = 0

    for ep in range(NUM_EP):
        s, r = ENV.reset(), 0
        agent.reset()
        ep_reward = 0
        optimizer.zero_grad()
        for ep_time in range(NUM_EP_STEP):
            a = agent.step(s, r, ep_time)
            s, r, done, info = ENV.step(a)

            ep_reward += r
            T += 1

            if done:
                loss = agent(done)
                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()
                break
            elif (ep_time+1) % TRAIN_INTERVAL == 0:
                # run additional step for bootstrap
                a = agent.step(s, r, ep_time)
                s, r, done, info = ENV.step(a)
                loss = agent(False)    # enable bootstrap regardless of done
                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()
                agent.reset(done)
                if done:    # episode sometimes finishes at the bootstrap step
                    break

        #print('Episode:', ep, 'Step:', T, 'Reward:', ep_reward)
        reward_log.append(ep_reward)


if __name__ == '__main__':
    agent = Merlin()
    optimizer = optim.Adam(agent.parameters(), lr=0.0001, eps=1e-9, betas=[0.9, 0.98])
    reward_log = []
    try:
        main()
    except:
        import traceback
        traceback.print_exc()

    # visualize learning history
    # if LOGGING:
    #     visualize_log(reward=reward_log, MBP_loss=agent.mbp_loss_log, policy_loss=agent.policy_loss_log)

    # # save the model
    # if SAVE_MODEL:
    #     serializers.save_npz(LOGDIR+'model.npz', agent)