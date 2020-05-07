"""
Implementation of the experiment file of proximal policy gradient baseline
@author : Bastien van Delft
"""

from rl_baselines.common.networks import CNNDeepmind_Multihead
from rl_baselines.envs.atari_wrappers import make_atari, wrap_deepmind
# from rl_baselines.baselines import PPO
from
from rl_baselines.common.networks.mlp import MLP_Multihead


import gym
import pickle
import numpy as np
import matplotlib.pyplot as plt

#env = make_atari("BreakoutNoFrameskip-v4")
#env = wrap_deepmind(env)
#env = gym.make("CartPole-v0")
env = gym.make("Pendulum-v0")

#env = wrap_deepmind(env)
#env = make_atari("PongNoFrameskip-v4")
#env = wrap_deepmind(env)
agent = PPO(env,
            MLP_Multihead,
            verbose=2,
            learning_rate=0.0004,#0.0002
            horizon = 200,
            epsilon = 0.3,#0.1
            output2_dim=1,
            coef_entropy = 0.01,
            num_envs = 16)


agent.learn(timesteps=100000000)

logdata = pickle.load(open("log_data.pkl",'rb'))
scores = np.array(logdata['Episode_score'])
plt.plot(scores[:,1], scores[:,0])
plt.show()
