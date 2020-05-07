"""
Implementation of the experiment file of Vanilla policy gradient baseline
@author : Benoit ROBAGLIA
"""

from rl_baselines.common.networks import MLP_Multihead
from rl_baselines.baselines import VPG
import gym
import pickle
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")

agent = VPG(env,
            MLP_Multihead,
            gamma=0.99,
            verbose=2,
            learning_rate=0.001)

agent.learn(timesteps=50000)

logdata = pickle.load(open("log_data.pkl",'rb'))
scores = np.array(logdata['Episode_score'])
plt.plot(scores[:,1], scores[:,0])
plt.show()