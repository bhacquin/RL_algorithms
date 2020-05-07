from rl_baselines.common.networks.networks_continuous_actions import QValueNetwork, PolicyNetwork
from rl_baselines.baselines.td3.td3 import TD3, GaussianExploration, OUNoise
import gym
import pickle
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("MountainCarContinuous-v0")
nb_steps = 20000
args_noise = [0.0, 0.15, 0.2, 0.2, 1000]

agent = TD3(env,
            PolicyNetwork,
            QValueNetwork,
            OUNoise,
            noise_clip=0.5,
            noise_std=0.2,
            args_noise=args_noise,
            replay_start_size=500,
            replay_buffer_size=1000000,
            update_target_frequency=4,
            gamma=0.99,
            polyak=1e-2,
            minibatch_size=64,
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
