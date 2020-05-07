"""
Implementation of the experiment file of proximal policy gradient baseline
@author : Bastien van Delft
"""

from rl_baselines.common.networks import CNNDeepmind_Multihead
from rl_baselines.envs.atari_wrappers import make_atari, wrap_deepmind
# from rl_baselines.baselines import PPO
from ppo_new_ongoing_bis import PPO
from rl_baselines.common.networks.mlp import MLP_Multihead
from carracing_wrapper import SkipEnv, FrameStack

import gym
import pickle
import numpy as np
import matplotlib.pyplot as plt

#env = make_atari("BreakoutNoFrameskip-v4")
#env = wrap_deepmind(env)
env = make_atari("PongNoFrameskip-v4")
# env = wrap_deepmind(env)
#env = gym.make("CarRacing-v0")
# env = gym.make("LunarLander-v2")
# env = gym.make('MountainCarContinuous-v0')

#env = wrap_deepmind(env)
#env = make_atari("PongNoFrameskip-v4")
#env = wrap_deepmind(env)

agent = PPO(env,
            network=CNNDeepmind_Multihead,
            wrapper=wrap_deepmind,
            verbose=False,
            num_epoch=5,
            learning_rate=0.003,#0.0002
            horizon = 128,
            tau=0.95,
            epsilon = 0.2,#0.1
            coef_entropy = 0.01,
            num_envs = 2,
            clip_grad_norm = 0.5,
            n_mini_batches=8,
            normalize_state=False,
            normalize_reward=True
            )


agent.learn(epochs=1000, n_steps=2000)

#logdata = pickle.load(open("log_data.pkl",'rb'))
#scores = np.array(logdata['Episode_score'])
# plt.plot(scores[:,1], scores[:,0])
# plt.show()
