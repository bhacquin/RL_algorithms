#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:18:48 2019

@author: maxime

"""

import rl_baselines.envs.timeseries.core as timeseries
import rl_baselines.common.networks.mlp as mlp
import rl_baselines.baselines as dqn

import pickle
import matplotlib.pyplot as plt
import numpy as np

env = timeseries.make("Triangle")
nb_steps = 10000

agent = dqn.DQN( env,
                 mlp.MLP,
                 replay_start_size=1000,
                 replay_buffer_size=1000,
                 gamma=0.9,
                 update_target_frequency=100,
                 minibatch_size=32,
                 learning_rate=1e-4,
                 update_frequency=4,
                 initial_exploration_rate=1,
                 final_exploration_rate=0.05,
                 final_exploration_step=int(0.1*nb_steps),
                 adam_epsilon=1e-8,
                 logging=True)


agent.learn(timesteps=nb_steps, verbose=True)

data = pickle.load(open("log_data.pkl",'rb'))
scores = np.array(data['Episode_score'])
plt.plot(scores[:,1],scores[:,0])
plt.show()

agent.save()


