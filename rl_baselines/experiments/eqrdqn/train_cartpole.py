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

from rl_baselines.baselines import EQRDQN
from rl_baselines.common.networks.mlp import MLP_Multihead


def run(
    *,
    n_quantiles: int = 50,
    kappa: int = 0,
    prior: int = 0.01,
    nb_steps: int = 20000,
    replay_start_size: int = 1000,
    replay_buffer_size: int = 50000,
    gamma: float = 0.99,
    update_target_frequency: int = 500,
    minibatch_size: int = 32,
    learning_rate: float = 1e-3,
    update_frequency: int = 4,
    adam_epsilon: float = 1e-8,
    log_folder: str = 'cartpole',
    logging: bool = True,
):
    print(locals())

    env = gym.make("CartPole-v0")

    eqrdqn_args = dict(locals())
    del eqrdqn_args['nb_steps']

    agent = EQRDQN(
        network=MLP_Multihead,
        **eqrdqn_args
    )

    agent.learn(timesteps=nb_steps, verbose=True)
    agent.save()

    if eqrdqn_args['log_folder'] is None:
        filename = 'log_data.pkl'
    else:
        filename = eqrdqn_args['log_folder'] + '/log_data.pkl'
    logdata = pickle.load(open(filename, 'rb'))
    scores = np.array(logdata['Episode_score'])
    plt.plot(scores[:, 1], scores[:, 0])
    plt.show()


if __name__ == "__main__":
    import clize

    clize.run(run)
