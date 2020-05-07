from rl_baselines.common.networks.networks_continuous_actions import QValueNetwork, PolicyNetworkSAC, ValueNetwork
from rl_baselines.baselines import SAC
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


nb_steps = 20000

agent = SAC(env,
        PolicyNetworkSAC,
        ValueNetwork,
        QValueNetwork,
        alpha=0.2,
        replay_start_size=500,
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
logdata = pickle.load(open("log_data.pkl",'rb'))
scores = np.array(logdata['Episode_score'])
plt.plot(scores[:,1],scores[:,0])
plt.show()
