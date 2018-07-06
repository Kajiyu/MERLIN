#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch as T
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from merlin.constants import *
from merlin.z_network import Z_network


def z_network_test(z_network):
    pass


if __name__ == '__main__':
    z_network = Z_network()