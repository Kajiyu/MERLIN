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
from merlin.deeplstm import DeepLSTM


if __name__ == '__main__':
    lstm = DeepLSTM(10, 30)
    x = T.ones(3, 1, 10)
    y = lstm(x)
    print ("output size: ", y.size())
    print ("output value: ", y)