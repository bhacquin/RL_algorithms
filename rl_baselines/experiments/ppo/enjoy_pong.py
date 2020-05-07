import torch
from pathlib import Path

from rl_baselines.common.networks import CNNDeepmind_Multihead
from rl_baselines.envs.atari_wrappers import make_atari, wrap_deepmind
from rl_baselines.baselines import PPO
from tqdm import tqdm

import gym
import pickle
import numpy as np
import matplotlib.pyplot as plt
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#env = make_atari("BreakoutNoFrameskip-v4")
#env = wrap_deepmind(env)

env = make_atari("PongNoFrameskip-v4")
env = wrap_deepmind(env)
agent = PPO(env,
            CNNDeepmind_Multihead,
            verbose=2,
            learning_rate=0.0001,
            render=True,
            x=1)
#
agent.load('Pong_network.pt')
for i in tqdm(range(10)):
    done = False
    state = torch.tensor(agent.env.reset(), dtype=torch.float, device=device)
    score = 0
    while not done:
        if agent.render:
            agent.env.render()
        action = agent.predict(state)
        next_state, reward, done, _ = agent.env.step(action)
        score += reward
        next_state = torch.tensor(next_state, dtype=torch.float, device=device)
        state = next_state
    print("score : {}".format(score))
agent.learn(episodes=nb_steps)
plt.plot(agent.get_score_history())
plt.show()
agent.save("DQN_Cartpole")