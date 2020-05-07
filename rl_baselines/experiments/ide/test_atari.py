#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:18:48 2019

@author: maxime

"""
import pickle
import torch
import time
import random

import gym
import matplotlib.pyplot as plt
import numpy as np

from rl_baselines.baselines import IDE
from rl_baselines.common.networks import CNNDeepmind_Threehead

from rl_baselines.envs.atari_wrappers import make_atari, wrap_deepmind

game_scores = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = make_atari("AlienNoFrameskip-v4",noop=True)
env = wrap_deepmind(env,clip_rewards=False,episode_life=False)

agent = IDE(env,CNNDeepmind_Threehead,n_quantiles=200)


for i in range(50):

    filename = "network_" + str((i+1)*250000) + ".pth"

    agent.load('alien/'+filename)

    score = 0
    scores = []
    total_timesteps = 0
    while total_timesteps < 125000:
        done = False
        obs = env.reset()
        lives = env.unwrapped.ale.lives()
        this_episode_time = 0
        while not done:
            if env.unwrapped.ale.lives() < lives:
                lives = env.unwrapped.ale.lives()
                if env.unwrapped.get_action_meanings()[1] == 'FIRE':
                    obs, rew, done, info = env.step(1)
            else:
                action = agent.predict(torch.FloatTensor(obs).to(device))
                obs, rew, done, info = env.step(action)
            score += rew
            this_episode_time += 1
            total_timesteps += 1

            if this_episode_time == 27000:
                done = True

            if done:
                #print(score)
                scores.append(score)
                score = 0
                i += 1

    print(np.mean(scores))
    print(np.std(scores))
    game_scores.append(np.mean(scores))

pickle.dump(game_scores, open( "alien_scores", "wb" ) )