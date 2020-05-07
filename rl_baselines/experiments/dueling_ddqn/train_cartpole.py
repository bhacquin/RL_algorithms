#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import gym

from rl_baselines.common.networks import MLP
from rl_baselines.baselines import DuelingDDQN


env = gym.make("CartPole-v0")
network = MLP(env, 64)
nb_steps = 100000

agent = DuelingDDQN(
    env,
    network,
    replay_start_size=1000, # todo
    replay_buffer_size=50000,
    gamma=0.99,
    update_evaluation_net_frequency=500,
    minibatch_size=32,
    learning_rate=1e-4,
    update_frequency=1,
    initial_exploration_rate=1,
    final_exploration_rate=0.02,
    final_exploration_step=int(0.1*nb_steps),
    adam_epsilon=1e-8)

agent.learn(timesteps=nb_steps, verbose=True)

plt.plot(agent.get_score_history())
plt.show()

agent.save()


