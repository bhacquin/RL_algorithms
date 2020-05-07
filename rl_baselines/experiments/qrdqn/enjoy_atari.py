#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:18:48 2019

@author: maxime

"""
import torch
import time

from rl_baselines.common.networks import CNNDeepmind
from rl_baselines.baselines import QRDQN
from rl_baselines.envs.atari_wrappers import make_atari, wrap_deepmind

import pickle
import numpy as np
import matplotlib.pyplot as plt

env = make_atari("BreakoutNoFrameskip-v4",noop=False)
env = wrap_deepmind(env,clip_rewards=False,episode_life=False)
#env.unwrapped.game_mode = 12

nb_steps = 50000000

agent = QRDQN( env,
                 CNNDeepmind,
                 n_quantiles=200,
                 kappa=0,
                 replay_start_size=50000, # todo
                 replay_buffer_size=1000000,
                 gamma=0.99,
                 update_target_frequency=10000,
                 minibatch_size=32,
                 learning_rate=2.5e-4,
                 initial_exploration_rate=1,
                 final_exploration_rate=0.01,
                 final_exploration_step=1000000,
                 adam_epsilon=0.01/32,
                 logging=True)

agent.load('results/2019-05-09/best_breakout.pth')

obs = env.reset()
score = 0
scores = []
lives = env.unwrapped.ale.lives()
for i in range(1000000):
    if env.unwrapped.ale.lives() < lives:
        lives = env.unwrapped.ale.lives()
        if env.unwrapped.get_action_meanings()[1] == 'FIRE':
            obs, rew, done, info = env.step(1)
    else:
        if np.random.uniform() < 0.01:
            action = np.random.choice(agent.env.action_space.n)
        else:
            action = agent.predict(torch.FloatTensor(obs))
        obs, rew, done, info = env.step(action)
    score += rew
    env.render()
    #time.sleep(0.02)
    if done:
        obs = env.reset()
        lives = env.unwrapped.ale.lives()
        print(score)
        scores.append(score)
        score=0

print(np.mean(scores))