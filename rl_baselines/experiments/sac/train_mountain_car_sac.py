from rl_baselines.common.networks.networks_continuous_actions import QValueNetwork, PolicyNetworkSAC, ValueNetwork
from rl_baselines.baselines import SAC
import gym
import pickle
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("MountainCarContinuous-v0")
nb_steps = 20000

agent = SAC(env,
        PolicyNetworkSAC,
        ValueNetwork,
        QValueNetwork,
        alpha=0.7,
        replay_start_size=5000,
        replay_buffer_size=1000000,
        update_target_frequency=2,
        gamma=0.99,
        polyak=1e-2,
        minibatch_size=64,
        policy_learning_rate=1e-3,
        value_learning_rate=1e-3,
        soft_q_learning_rate=1e-3,
        update_frequency=1,
        adam_epsilon=1e-8,
        logging=True)

agent.learn(timesteps=nb_steps, verbose=True)
# agent.save()

logdata = pickle.load(open("log_data.pkl",'rb'))
scores = np.array(logdata['Episode_score'])
plt.plot(scores[:,1],scores[:,0])
plt.show()
