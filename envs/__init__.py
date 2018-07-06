#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gym.envs.registration import register

register(
    id='Memory-v0',
    entry_point='envs.memory:Memory'
)