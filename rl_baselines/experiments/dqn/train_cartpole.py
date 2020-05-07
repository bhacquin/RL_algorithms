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

from rl_baselines.baselines import DQN
from rl_baselines.common.networks.mlp import MLP


def run(
    *,
    nb_steps: int = 20000,
    replay_start_size: int = 1000,
    replay_buffer_size: int = 50000,
    gamma: float = 0.99,
    update_target_frequency: int = 500,
    minibatch_size: int = 32,
    learning_rate: float = 1e-3,
    update_frequency: int = 4,
    initial_exploration_rate: float = 1,
    final_exploration_rate: float = 0.02,
    final_exploration_step: int = 5000,
    adam_epsilon: float = 1e-8,
    logging: bool = True,
    loss: str = "mse",
):
    print(locals())

    env = gym.make("CartPole-v0")

    if final_exploration_step is None:
        final_exploration_step = int(0.1 * nb_steps)

    dqn_args = dict(locals())
    del dqn_args['nb_steps']

    agent = DQN(
        network=MLP,
        **dqn_args
    )

    agent.learn(timesteps=nb_steps, verbose=True)
    agent.save()

    logdata = pickle.load(open("log_data.pkl", 'rb'))
    scores = np.array(logdata['Episode_score'])
    plt.plot(scores[:, 1], scores[:, 0])
    plt.show()


if __name__ == "__main__":
    import clize

    clize.run(run)
