from rl_baselines.common.networks.networks_continuous_actions import QValueNetwork, PolicyNetwork
from rl_baselines.baselines.ddpg.ddpg import DDPG
import gym
import pickle
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("LunarLanderContinuous-v2")
nb_steps = 20000000


agent = DDPG(env,
             PolicyNetwork,
             QValueNetwork,
             replay_start_size=500,
             replay_buffer_size=1000000,
             gamma=0.99,
             polyak=1e-2,
             minibatch_size=64,
             policy_learning_rate=1e-3,
             value_learning_rate=1e-3,
             update_frequency=1,
             adam_epsilon=1e-8,
             logging=True)

agent.learn(timesteps=nb_steps, verbose=True)
# agent.save()

logdata = pickle.load(open("log_data.pkl",'rb'))
scores = np.array(logdata['Episode_score'])
plt.plot(scores[:,1],scores[:,0])
plt.show()
