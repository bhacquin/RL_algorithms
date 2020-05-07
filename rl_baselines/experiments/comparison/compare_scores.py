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

from rl_baselines.baselines import EQRDQN
from rl_baselines.baselines import QRDQN
from rl_baselines.common.networks import CNNDeepmind_Multihead
from rl_baselines.common.networks import CNNDeepmind

from rl_baselines.envs.atari_wrappers import make_atari, wrap_deepmind

game_scores = []


env = make_atari("BreakoutNoFrameskip-v4",noop=True)
env = wrap_deepmind(env,clip_rewards=False,episode_life=False)
env.unwrapped.game_mode = 12

agent = EQRDQN(env,CNNDeepmind_Multihead,n_quantiles=200)
agent.load('../eqrdqn/results/2019-05-09/best_breakout.pth')

score = 0
scores = []
for i in range(100):
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
            if np.random.uniform() < 0.01:
                action = np.random.choice(agent.env.action_space.n)
            else:
                action = agent.predict(torch.FloatTensor(obs))
            obs, rew, done, info = env.step(action)
        score += rew
        this_episode_time += 1
        #env.render()
        #time.sleep(0.02)

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


env = make_atari("BreakoutNoFrameskip-v4",noop=True)
env = wrap_deepmind(env,clip_rewards=False,episode_life=False)
env.unwrapped.game_mode = 12

agent = QRDQN(env,CNNDeepmind,n_quantiles=200)
agent.load('../qrdqn/results/2019-05-09/best_breakout.pth')

score = 0
scores = []
for i in range(100):
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
            if np.random.uniform() < 0.01:
                action = np.random.choice(agent.env.action_space.n)
            else:
                action = agent.predict(torch.FloatTensor(obs))
            obs, rew, done, info = env.step(action)
        score += rew
        this_episode_time += 1
        #env.render()
        #time.sleep(0.02)

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

"""
env = make_atari("AlienNoFrameskip-v0",noop=True)
env = wrap_deepmind(env,clip_rewards=False,episode_life=False)

agent = EQRDQN(env,CNNDeepmind_Multihead,n_quantiles=200)
agent.load('../eqrdqn/results/2019-05-07/best_alien.pth')

score = 0
scores = []
for i in range(30):
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
            if np.random.uniform() < 0.001:
                action = np.random.choice(agent.env.action_space.n)
            else:
                action = agent.predict(torch.FloatTensor(obs))
            obs, rew, done, info = env.step(action)
        score += rew
        this_episode_time += 1
        #env.render()
        #time.sleep(0.02)

        if this_episode_time == 27000:
            done = True

        if done:
            #print(score)
            scores.append(score)
            score = 0
            i += 1

print(np.mean(scores))
game_scores.append(np.mean(scores))

env = make_atari("AmidarNoFrameskip-v0",noop=True)
env = wrap_deepmind(env,clip_rewards=False,episode_life=False)

agent = EQRDQN(env,CNNDeepmind_Multihead,n_quantiles=200)
agent.load('../eqrdqn/results/2019-05-07/best_amidar.pth')

score = 0
scores = []
for i in range(30):
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
            if np.random.uniform() < 0.001:
                action = np.random.choice(agent.env.action_space.n)
            else:
                action = agent.predict(torch.FloatTensor(obs))
            obs, rew, done, info = env.step(action)
        score += rew
        this_episode_time += 1
        #env.render()
        #time.sleep(0.02)

        if this_episode_time == 27000:
            done = True

        if done:
            #print(score)
            scores.append(score)
            score = 0
            i += 1

print(np.mean(scores))
game_scores.append(np.mean(scores))

env = make_atari("AssaultNoFrameskip-v0",noop=True)
env = wrap_deepmind(env,clip_rewards=False,episode_life=False)

agent = EQRDQN(env,CNNDeepmind_Multihead,n_quantiles=200)
agent.load('../eqrdqn/results/2019-05-07/best_assault.pth')

score = 0
scores = []
for i in range(30):
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
            if np.random.uniform() < 0.001:
                action = np.random.choice(agent.env.action_space.n)
            else:
                action = agent.predict(torch.FloatTensor(obs))
            obs, rew, done, info = env.step(action)
        score += rew
        this_episode_time += 1
        #env.render()
        #time.sleep(0.02)

        if this_episode_time == 27000:
            done = True

        if done:
            #print(score)
            scores.append(score)
            score = 0
            i += 1

print(np.mean(scores))
game_scores.append(np.mean(scores))

env = make_atari("AsterixNoFrameskip-v0",noop=True)
env = wrap_deepmind(env,clip_rewards=False,episode_life=False)

agent = EQRDQN(env,CNNDeepmind_Multihead,n_quantiles=200)
agent.load('../eqrdqn/results/2019-05-07/best_asterix.pth')

score = 0
scores = []
for i in range(30):
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
            if np.random.uniform() < 0.001:
                action = np.random.choice(agent.env.action_space.n)
            else:
                action = agent.predict(torch.FloatTensor(obs))
            obs, rew, done, info = env.step(action)
        score += rew
        this_episode_time += 1
        #env.render()
        #time.sleep(0.02)

        if this_episode_time == 27000:
            done = True

        if done:
            #print(score)
            scores.append(score)
            score = 0
            i += 1

print(np.mean(scores))
game_scores.append(np.mean(scores))

env = make_atari("AlienNoFrameskip-v0",noop=True)
env = wrap_deepmind(env,clip_rewards=False,episode_life=False)

agent = QRDQN(env,CNNDeepmind,n_quantiles=200)
agent.load('../qrdqn/results/2019-05-07/best_alien.pth')

score = 0
scores = []
for i in range(30):
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
            if np.random.uniform() < 0.001:
                action = np.random.choice(agent.env.action_space.n)
            else:
                action = agent.predict(torch.FloatTensor(obs))
            obs, rew, done, info = env.step(action)
        score += rew
        this_episode_time += 1
        #env.render()
        #time.sleep(0.02)

        if this_episode_time == 27000:
            done = True

        if done:
            #print(score)
            scores.append(score)
            score = 0
            i += 1

print(np.mean(scores))
game_scores.append(np.mean(scores))

env = make_atari("AmidarNoFrameskip-v0",noop=True)
env = wrap_deepmind(env,clip_rewards=False,episode_life=False)

agent = QRDQN(env,CNNDeepmind,n_quantiles=200)
agent.load('../qrdqn/results/2019-05-07/best_amidar.pth')

score = 0
scores = []
for i in range(30):
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
            if np.random.uniform() < 0.001:
                action = np.random.choice(agent.env.action_space.n)
            else:
                action = agent.predict(torch.FloatTensor(obs))
            obs, rew, done, info = env.step(action)
        score += rew
        this_episode_time += 1
        #env.render()
        #time.sleep(0.02)

        if this_episode_time == 27000:
            done = True

        if done:
            #print(score)
            scores.append(score)
            score = 0
            i += 1

print(np.mean(scores))
game_scores.append(np.mean(scores))

env = make_atari("AssaultNoFrameskip-v0",noop=True)
env = wrap_deepmind(env,clip_rewards=False,episode_life=False)

agent = QRDQN(env,CNNDeepmind,n_quantiles=200)
agent.load('../qrdqn/results/2019-05-07/best_assault.pth')

score = 0
scores = []
for i in range(30):
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
            if np.random.uniform() < 0.001:
                action = np.random.choice(agent.env.action_space.n)
            else:
                action = agent.predict(torch.FloatTensor(obs))
            obs, rew, done, info = env.step(action)
        score += rew
        this_episode_time += 1
        #env.render()
        #time.sleep(0.02)

        if this_episode_time == 27000:
            done = True

        if done:
            #print(score)
            scores.append(score)
            score = 0
            i += 1

print(np.mean(scores))
game_scores.append(np.mean(scores))

env = make_atari("AsterixNoFrameskip-v0",noop=True)
env = wrap_deepmind(env,clip_rewards=False,episode_life=False)

agent = QRDQN(env,CNNDeepmind,n_quantiles=200)
agent.load('../qrdqn/results/2019-05-07/best_asterix.pth')

score = 0
scores = []
for i in range(30):
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
            if np.random.uniform() < 0.001:
                action = np.random.choice(agent.env.action_space.n)
            else:
                action = agent.predict(torch.FloatTensor(obs))
            obs, rew, done, info = env.step(action)
        score += rew
        this_episode_time += 1
        #env.render()
        #time.sleep(0.02)

        if this_episode_time == 27000:
            done = True

        if done:
            #print(score)
            scores.append(score)
            score = 0
            i += 1

print(np.mean(scores))
game_scores.append(np.mean(scores))

print(game_scores)

"""