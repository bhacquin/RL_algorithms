#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:18:48 2019

@author: maxime

"""

import rl_baselines.envs.timeseries.core as timeseries

from rl_baselines.common.networks import MLP_Multihead
from rl_baselines.baselines import VPG
import gym
import pickle
import numpy as np
import matplotlib.pyplot as plt

env = timeseries.make("Triangle")

agent = VPG(env,
            MLP_Multihead,
            gamma=0.99,
            verbose=True,
            learning_rate=0.001)

agent.learn(timesteps=100000)

logdata = pickle.load(open("log_data.pkl",'rb'))
scores = np.array(logdata['Episode_score'])
plt.plot(scores[:,1], scores[:,0])
plt.show()


