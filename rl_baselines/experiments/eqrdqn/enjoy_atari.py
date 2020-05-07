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

from rl_baselines.baselines import EQRDQN
from rl_baselines.common.networks import CNNDeepmind_Multihead

from rl_baselines.envs.atari_wrappers import make_atari, wrap_deepmind

env = make_atari("BreakoutNoFrameskip-v4",noop=False)
env = wrap_deepmind(env,clip_rewards=False,episode_life=False)

agent = EQRDQN(env,CNNDeepmind_Multihead,n_quantiles=200)

agent.load('results/2019-04-29/Breakout/network.pth')

"""
obs = env.reset()
returns = 0
for i in range(10000):
    action = agent.predict(torch.FloatTensor(obs))
    obs, rew, done, info = env.step(action)
    env.render()
    #time.sleep(0.02)
    returns += rew
    if done:
        obs = env.reset()
        print(returns)
        returns = 0

"""

obs = env.reset()
returns = 0
for i in range(10000):
    net1,net2 = agent.network(torch.FloatTensor(obs))
    net1 = net1.view(agent.env.action_space.n,agent.n_quantiles)
    net2 = net2.view(agent.env.action_space.n,agent.n_quantiles)
    
    """
    plt.cla()
    #plt.axis([0,200,0,6])
    pred1 = (np.array(net1[0,:].detach()) + np.array(net2[0,:].detach()))/2
    pred2 = (np.array(net1[1,:].detach()) + np.array(net2[1,:].detach()))/2
    plt.plot(np.array(net1[2,:].detach()), 'r', label="up")
    plt.plot(np.array(net2[2,:].detach()), 'r', label="up")
    plt.plot(np.array(net1[3,:].detach()), 'g', label="down")
    plt.plot(np.array(net2[3,:].detach()), 'g', label="down")
    plt.legend()
    plt.draw()
    plt.pause(0.01) 
    """
    
    uncertainty = torch.sqrt(torch.mean((net1-net2)**2,dim=1)/2).detach()
    means = torch.mean((net1+net2)/2,dim=1).detach()
    if np.random.uniform() < 0.001:
        action = np.random.choice(agent.env.action_space.n)
    else:
        action = agent.predict(torch.FloatTensor(obs))
    obs, rew, done, info = env.step(action)
    env.render()
    #time.sleep(0.2)
    returns += rew
    #print(action, "means",means,"uncertainties",uncertainty)
    if done:
        obs = env.reset()
        print(returns)
        returns = 0

