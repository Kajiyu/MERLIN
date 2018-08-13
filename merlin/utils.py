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


def make_batch(*xs):
    """ xs:     not batched variables.
        return: list of batched variables.
    """
    return [T.from_numpy(x.reshape(1,-1)) if type(x)==Variable else T.from_numpy(np.array(x, dtype=np.float32).reshape(1,-1)) for x in xs]