#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:18:48 2019

@author: maxime

"""
from rl_baselines.common.networks import CNNDeepmind_Threehead
from rl_baselines.baselines import IDE
from rl_baselines.envs.atari_wrappers import make_atari, wrap_deepmind

import pickle
import numpy as np
import matplotlib.pyplot as plt

env = make_atari("AlienNoFrameskip-v4",noop=True)
env = wrap_deepmind(env, episode_life=True)

nb_steps = 12500000

agent = IDE( env,
                 CNNDeepmind_Threehead,
                 n_quantiles=200,
                 kappa=0,
                 prior = 0.0001,
                 replay_start_size=50000,
                 replay_buffer_size=1000000,
                 gamma=0.99,
                 update_target_frequency=10000,
                 minibatch_size=32,
                 learning_rate=5e-5,
                 adam_epsilon=0.01/32,
                 update_frequency=4,
                 log_folder='alien',
                 logging=True)


agent.learn(timesteps=nb_steps, verbose=True)
agent.save()
