from rl_baselines.common.networks.networks_continuous_actions import QValueNetwork, PolicyNetwork
from rl_baselines.baselines.ddpg.ddpg import DDPG, OUNoise
import gym
import pickle
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("MountainCarContinuous-v0")
nb_steps = 20000
args_noise = [0.0, 0.15, 0.2, 0.2, 1000]

agent = DDPG(env,
             PolicyNetwork,
             QValueNetwork,
             OUNoise,
             args_noise=args_noise,
             replay_start_size=500,
             replay_buffer_size=1000000,
             gamma=0.99,
             polyak=1e-2,
             minibatch_size=32,
             policy_learning_rate=1e-3,
             value_learning_rate=1e-3,
             update_frequency=4,
             adam_epsilon=1e-8,
             logging=True)

agent.learn(timesteps=nb_steps, verbose=True)
# agent.save()

logdata = pickle.load(open("log_data.pkl",'rb'))
scores = np.array(logdata['Episode_score'])
plt.plot(scores[:,1],scores[:,0])
plt.show()
