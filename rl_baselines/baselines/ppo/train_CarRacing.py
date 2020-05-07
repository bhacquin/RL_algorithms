"""
Implementation of the experiment file of proximal policy gradient baseline
@author : Bastien van Delft
"""

from rl_baselines.common.networks import CNNDeepmind_Multihead
from rl_baselines.envs.atari_wrappers import make_atari, wrap_deepmind
# from rl_baselines.baselines import PPO
from ppo_new_ongoing import PPO
from rl_baselines.common.networks.mlp import MLP_Multihead
from carracing_wrapper import SkipEnv, FrameStack
import torch
import gym
import pickle
import numpy as np
import matplotlib.pyplot as plt


env = gym.make("CarRacing-v0")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def wrapper(env):
    env = SkipEnv(env,skip=5)
    # env = ChangeColorEnv(env,color_range)
    env = FrameStack(env,2)
    return env


agent = PPO(env,
            wrapper=wrapper,
            verbose=2,
            learning_rate=0.0002,#0.0002
            horizon = 1000,
            tau=0.95,
            epsilon = 0.1,#0.1
            coef_entropy = 0.01,
            num_envs = 2,
            clip_grad_norm = 0.5,
            n_mini_batches=10,
            v_clip_range = 0.5,
            normalize_state=False,
            normalize_reward=True,
            device = device
            )


agent.learn(epochs=100, n_steps=1000)

#logdata = pickle.load(open("log_data.pkl",'rb'))
#scores = np.array(logdata['Episode_score'])
# plt.plot(scores[:,1], scores[:,0])
# plt.show()
