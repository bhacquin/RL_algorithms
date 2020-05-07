#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:18:48 2019

@author: maxime

"""
import pickle
import torch
import time

import gym
import matplotlib.pyplot as plt
import numpy as np

from rl_baselines.baselines import DQN
from rl_baselines.common.networks.mlp import MLP

env = gym.make("CartPole-v0")

agent = DQN(env,MLP)

agent.load('network.pth')


obs = env.reset()
returns = 0
for i in range(10000):
    action = agent.predict(torch.FloatTensor(obs))
    obs, rew, done, info = env.step(action)
    env.render()
    time.sleep(0.02)
    returns += rew
    if done:
        obs = env.reset()
        print(returns)
        returns = 0