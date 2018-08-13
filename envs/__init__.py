#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym.envs.registration import register

register(
    id='Memory-v0',
    entry_point='envs.memory:Memory'
)