#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:18:48 2019

@author: maxime

"""
import pickle

import gym
import matplotlib.pyplot as plt
import numpy as np

from rl_baselines.baselines import DDQN, DuelingMLP
from rl_baselines.common.networks import MLP

env = gym.make("CartPole-v0")
nb_steps = 8000

agent = DDQN(
    env,
    DuelingMLP,
    replay_start_size=1000,  # todo
    replay_buffer_size=50000,
    gamma=0.99,
    update_target_frequency=500,
    minibatch_size=32,
    learning_rate=1e-4,
    update_frequency=1,
    initial_exploration_rate=1,
    final_exploration_rate=0.02,
    final_exploration_step=int(0.1 * nb_steps),
    adam_epsilon=1e-8,
    logging=True,
)


agent.learn(timesteps=nb_steps, verbose=True)
agent.save()

log_data = pickle.load(open('log_data.pkl', 'rb'))
scores = np.array(log_data['Episode_score'])
plt.plot(scores[:, 1], scores[:, 0])
plt.show()
