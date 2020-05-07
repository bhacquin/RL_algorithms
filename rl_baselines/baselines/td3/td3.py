# Twin Delayed DDPG
# @Benoit

import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_baselines.common import ReplayBuffer
from rl_baselines.common.logger import Logger
from rl_baselines.common.utils import set_global_seed


# Noise Objects

class GaussianExploration(object):
    def __init__(self, action_space, max_sigma=1.0, min_sigma=1.0, decay_period=1000000):
        self.low = action_space.low
        self.high = action_space.high
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period

    def get_action(self, action, t=0):
        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        action = action + np.random.normal(size=len(action)) * sigma
        return np.clip(action, self.low, self.high)


class OUNoise:
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.state = None
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


class TD3:

    def __init__(self,
                 env,
                 policy_network,
                 value_network,
                 noise=OUNoise,
                 noise_clip=0.5,
                 noise_std=0.2,
                 args_noise=[],
                 replay_start_size=50000,
                 replay_buffer_size=1000000,
                 update_target_frequency=10,
                 gamma=0.99,
                 polyak=1e-2,
                 minibatch_size=32,
                 policy_learning_rate=1e-3,
                 value_learning_rate=1e-3,
                 update_frequency=4,
                 adam_epsilon=1e-8,
                 seed=None,
                 logging=False):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_clip = noise_clip
        self.noise_std = noise_std
        self.replay_start_size = replay_start_size
        self.replay_buffer_size = replay_buffer_size
        self.update_target_frequency = update_target_frequency
        self.gamma = gamma
        self.polyak = polyak
        self.minibatch_size = minibatch_size
        self.policy_learning_rate = policy_learning_rate
        self.value_learning_rate = value_learning_rate
        self.update_frequency = update_frequency
        self.adam_epsilon = adam_epsilon
        self.logging = logging

        self.env = env
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.seed = np.random.randint(0,1e6) if seed == None else seed
        set_global_seed(self.seed, self.env)


        # Init networks and load state dict
        self.policy_network = policy_network(self.env.observation_space, self.env.action_space, 256).to(self.device)
        self.target_policy_network = policy_network(self.env.observation_space, self.env.action_space, 256).to(self.device)
        self.target_policy_network.load_state_dict(self.policy_network.state_dict())

        self.value_network1 = value_network(self.env.observation_space, self.env.action_space, 256).to(self.device)
        self.target_value_network1 = value_network(self.env.observation_space, self.env.action_space, 256).to(self.device)
        self.target_value_network1.load_state_dict(self.value_network1.state_dict())

        self.value_network2 = value_network(self.env.observation_space, self.env.action_space, 256).to(self.device)
        self.target_value_network2 = value_network(self.env.observation_space, self.env.action_space, 256).to(self.device)
        self.target_value_network2.load_state_dict(self.value_network2.state_dict())

        # Set noise and optimizer
        self.noise = noise(self.env.action_space, *args_noise)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.policy_learning_rate, eps=self.adam_epsilon)
        self.value_optimizer1 = optim.Adam(self.value_network1.parameters(), lr=self.value_learning_rate, eps=self.adam_epsilon)
        self.value_optimizer2 = optim.Adam(self.value_network2.parameters(), lr=self.value_learning_rate, eps=self.adam_epsilon)

    def _update_target(self, network, target_network):

        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.polyak) + param.data * self.polyak)

    def act(self, state, is_training_ready, step=0):

        if is_training_ready:
            with torch.no_grad():
                action = self.policy_network(state.float().unsqueeze(0)).cpu().numpy()[0]
                action = self.noise.get_action(action, step)
                return action

        else:
            return self.env.action_space.sample()


    def learn(self, timesteps, verbose=False):

        if self.logging:
            logger = Logger()

        # On initialise l'état
        state = torch.as_tensor(self.env.reset())
        done = False
        score = 0
        ep_len = 0
        t1 = time.time()

        for timestep in range(timesteps):

            is_training_ready = (timestep >= self.replay_start_size)

            # On prend une action
            action = self.act(state.to(self.device), is_training_ready, step=ep_len)

            # On execute l'action dans l'environnement
            state_next, reward, done, _ = self.env.step(action)

            # On stock la transition dans le replay buffer
            action = torch.as_tensor(action, dtype=torch.float)
            reward = torch.as_tensor([reward], dtype=torch.float)
            done = torch.as_tensor([done], dtype=torch.float)
            state_next = torch.as_tensor(state_next)
            self.replay_buffer.add(state, action, reward, state_next, done)

            score += reward.item()
            ep_len += 1

            if done:
                # On réinitialise l'état
                if verbose:
                    print("Timestep : {}, score : {:0.2f}, Time : {:0.2f} s".format(timestep, score, time.time() - t1, 3))
                if self.logging:
                    logger.add_scalar('Episode_score', score, timestep)
                state = torch.as_tensor(self.env.reset())
                done = False
                score = 0
                ep_len = 0
                t1 = time.time()
            else:
                state = state_next

            if is_training_ready:

                # Update du réseau principal
                if timestep % self.update_frequency == 0:

                    # On sample un minibatche de transitions
                    transitions = self.replay_buffer.sample(self.minibatch_size, self.device)

                    # On s'entraine sur les transitions selectionnées
                    policy_loss, value_loss = self.train_step(transitions, timestep)
                    if self.logging:
                        logger.add_scalar('policy_loss', policy_loss, timestep)
                        logger.add_scalar('value_loss', value_loss, timestep)

                # Print de la loss
                # if (timestep % 100000) & (verbose):
                #     print(self.loss[-1])

        if self.logging:
            logger.save()

    def train_step(self, transitions, timestep):
        # Define the value criterion
        value_criterion = nn.MSELoss()

        states, actions, rewards, states_next, dones = transitions

        # update value functions
        next_action = self.target_policy_network(states_next.float())
        noise = torch.normal(torch.zeros(next_action.size()), self.noise_std).to(self.device)
        noise = torch.clamp(noise, - self.noise_clip, self.noise_clip)
        next_action += noise
        next_action = torch.clamp(next_action, self.env.action_space.low[0], self.env.action_space.high[0])

        with torch.no_grad():
            target_value1 = self.target_value_network1(states_next.float(), next_action)
            target_value2 = self.target_value_network2(states_next.float(), next_action)
            target_value = torch.min(target_value1, target_value2)


        expected_value = rewards + (1.0 - dones) * self.gamma * target_value

        value1 = self.value_network1(states.float(), actions)
        value2 = self.value_network2(states.float(), actions)

        with torch.no_grad():
            value_loss1 = value_criterion(value1, expected_value)
            value_loss2 = value_criterion(value2, expected_value)

        self.value_optimizer1.zero_grad()
        value_loss1.backward()
        self.value_optimizer1.step()
        self.value_optimizer2.zero_grad()
        value_loss2.backward()
        self.value_optimizer2.step()


        # Update policy and target networks
        if timestep % self.update_target_frequency == 0:
            policy_loss = self.value_network1(states.float(), self.policy_network(states.float()))
            policy_loss = - policy_loss.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        else:
            policy_loss = torch.tensor(np.nan)

        # Update target
        self.update_target(self.value_network1, self.target_value_network1)
        self.update_target(self.value_network2, self.target_value_network2)
        self.update_target(self.policy_network, self.target_policy_network)


        return policy_loss.item(), value_loss1.item()

    def save(self):
        torch.save(self.value_network1.state_dict(), 'value_network.pth')
        torch.save(self.policy_network.state_dict(), 'policy_network.pth')
