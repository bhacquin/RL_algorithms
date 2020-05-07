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

agent.load('results/2019-05-03/Breakout/network.pth')

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

returns = 0
done = False
obs = env.reset()
lives = env.unwrapped.ale.lives()
this_episode_time = 0
uncertainties = []
means = []
stds=[]
deaths = []
while not done:
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
    
    #means = torch.mean((net1+net2)/2,dim=1).detach()

    if env.unwrapped.ale.lives() < lives:
        lives = env.unwrapped.ale.lives()
        deaths.append(this_episode_time)
        if env.unwrapped.get_action_meanings()[1] == 'FIRE':
            obs, rew, done, info = env.step(1)
    else:
        if np.random.uniform() < 0.05:
            action = np.random.choice(agent.env.action_space.n)
        else:
            action = agent.predict(torch.FloatTensor(obs))
        obs, rew, done, info = env.step(action)

    
    uncertainty = torch.sqrt(torch.mean((net1-net2)**2,dim=1)/2).detach()
    mean = torch.mean((net1+net2)/2,dim=1).detach()
    std = torch.std((net1+net2)/2,dim=1).detach()
    plt.cla()
    uncertainties.append(10*uncertainty[action])
    means.append(mean[action])
    stds.append(std[action])
    plt.plot(uncertainties)
    plt.plot(means)
    plt.plot(stds)
    if deaths:
        for i in deaths:
            plt.scatter(i,0,c='r')
    plt.draw()
    plt.pause(0.01) 

    env.render()
    #time.sleep(0.2)
    returns += rew
    #print(action, "means",means,"uncertainties",uncertainty)
    this_episode_time += 1
    if this_episode_time == 27000:
        done = True

    if done:
        print(returns)
        time.sleep(5)

with open('uncertainties', 'wb') as output:
    pickle.dump(uncertainties, output, pickle.HIGHEST_PROTOCOL)