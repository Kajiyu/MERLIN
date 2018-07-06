#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch as T
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from merlin.constants import *
from merlin.memory import Memory


def read_test(memory):
    print("Memory Reading Test: ")
    _k = T.ones(1, M_DIM*Kr)
    _b = T.eye(Kr)[0].view(1, -1)
    print("k tensor: ", _k)
    print("b tensor: ", _b)
    print(memory.read(_k, _b))


def write_test(memory):
    print("Memory Writing Test: ")
    _z = T.ones(1, Z_DIM)
    _t = 9
    print("z tensor: ", _z)
    print("Time: ", _t)
    a_m = memory.write(_z, _t, debug=True)
    print("After Memory: ", a_m)


if __name__ == '__main__':
    memory = Memory()
    # reading test
    read_test(memory)
    # writing test
    write_test(memory)
    # reading test
    read_test(memory)
    # writing test
    write_test(memory)