"""
Implementation of the experiment file of Vanilla policy gradient baseline
@author : William Clements
"""

from rl_baselines.common.networks import CNNDeepmind_Multihead
from rl_baselines.envs.atari_wrappers import make_atari, wrap_deepmind
from rl_baselines.baselines import VPG

import gym
import pickle
import numpy as np
import matplotlib.pyplot as plt

env = make_atari("PongNoFrameskip-v4")
env = wrap_deepmind(env)

agent = VPG(env,
            CNNDeepmind_Multihead,
            gamma=0.99,
            verbose=2,
            learning_rate=0.001)

agent.learn(timesteps=50000)

logdata = pickle.load(open("log_data.pkl",'rb'))
scores = np.array(logdata['Episode_score'])
plt.plot(scores[:,1], scores[:,0])
plt.show()