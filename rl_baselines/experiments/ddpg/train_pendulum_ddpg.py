from rl_baselines.common.networks.networks_continuous_actions import QValueNetwork, PolicyNetwork
from rl_baselines.baselines.ddpg.ddpg import DDPG, OUNoise
import gym
import pickle
import matplotlib.pyplot as plt
import numpy as np


class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


env = NormalizedActions(gym.make("Pendulum-v0"))


nb_steps = 12000

agent = DDPG(env,
             PolicyNetwork,
             QValueNetwork,
             replay_start_size=32,
             replay_buffer_size=1000000,
             gamma=0.99,
             polyak=0.995,
             minibatch_size=32,
             policy_learning_rate=3e-4,
             value_learning_rate=3e-4,
             update_frequency=1,
             adam_epsilon=1e-8,
             logging=True)

agent.learn(timesteps=nb_steps, verbose=True)
logdata = pickle.load(open("log_data.pkl",'rb'))
scores = np.array(logdata['Episode_score'])
plt.plot(scores[:,1],scores[:,0])
plt.show()
