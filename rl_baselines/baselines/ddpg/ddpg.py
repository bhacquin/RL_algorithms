# Deep Deterministic Policy Gradient
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


class OUNoise:
    def __init__(self, action_space, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
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

    def get_action(self, action):
        ou_state = self.evolve_state()
        return np.clip(action + ou_state, self.low, self.high)


class DDPG:
    def __init__(
        self,
        env,
        policy_network,
        value_network,
        noise=OUNoise,
        args_noise=[],
        replay_start_size=50000,
        replay_buffer_size=1000000,
        gamma=0.99,
        polyak=0.995,
        minibatch_size=32,
        policy_learning_rate=1e-3,
        value_learning_rate=1e-3,
        update_frequency=4,
        adam_epsilon=1e-8,
        seed=None,
        logging=False,
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replay_start_size = replay_start_size
        self.replay_buffer_size = replay_buffer_size
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
        self.seed = np.random.randint(0, 1e6) if seed is None else seed
        set_global_seed(self.seed, self.env)

        # Init networks and load state dict
        spaces = [self.env.observation_space, self.env.action_space, 256]

        self.policy_network = policy_network(*spaces).to(self.device)
        self.target_policy_network = policy_network(*spaces).to(self.device)
        self.target_policy_network.load_state_dict(self.policy_network.state_dict())

        self.value_network = value_network(*spaces).to(self.device)
        self.target_value_network = value_network(*spaces).to(self.device)
        self.target_value_network.load_state_dict(self.value_network.state_dict())

        # Set noise and optimizer
        self.noise = noise(self.env.action_space, *args_noise)
        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(), lr=self.policy_learning_rate, eps=self.adam_epsilon
        )
        self.value_optimizer = optim.Adam(
            self.value_network.parameters(), lr=self.value_learning_rate, eps=self.adam_epsilon
        )

    @torch.no_grad()
    def act(self, state, is_training_ready=True):
        if is_training_ready:
            action = self.policy_network(state.float().unsqueeze(0)).cpu().numpy()[0]
            action = self.noise.get_action(action)
            return action
        else:
            return self.env.action_space.sample()

    def learn(self, timesteps, verbose=False):
        if self.logging:
            logger = Logger()

        # On initialise l'état
        state = torch.as_tensor(self.env.reset())
        score = 0
        t1 = time.time()

        for timestep in range(timesteps):

            is_training_ready = timestep >= self.replay_start_size

            # On prend une action
            action = self.act(state.to(self.device).float(), is_training_ready=is_training_ready)
            # On execute l'action dans l'environnement
            state_next, reward, done, _ = self.env.step(action)

            # On stock la transition dans le replay buffer
            action = torch.as_tensor(action, dtype=torch.float)
            reward = torch.as_tensor([reward], dtype=torch.float)
            done = torch.as_tensor([done], dtype=torch.float)
            state_next = torch.as_tensor(state_next)
            self.replay_buffer.add(state, action, reward, state_next, done)

            score += reward.item()

            if done:
                # On réinitialise l'état
                if verbose:
                    print("Timestep : {}, score : {}, Time : {} s".format(timestep, score, round(time.time() - t1, 3)))
                if self.logging:
                    logger.add_scalar('Episode_score', score, timestep)
                state = torch.as_tensor(self.env.reset())
                score = 0
                t1 = time.time()
            else:
                state = state_next

            if is_training_ready:

                # Update du réseau principal
                if timestep % self.update_frequency == 0:

                    # On sample un minibatche de transitions
                    transitions = self.replay_buffer.sample(self.minibatch_size, self.device)

                    # On s'entraine sur les transitions selectionnées
                    policy_loss, value_loss = self.train_step(transitions)
                    if self.logging:
                        logger.add_scalar('policy_loss', policy_loss, timestep)
                        logger.add_scalar('value_loss', value_loss, timestep)

        if self.logging:
            logger.save()

    def train_step(self, transitions):
        # Define the value criterion
        value_criterion = nn.MSELoss()

        states, actions, rewards, states_next, dones = transitions
        states = states.float()
        states_next = states_next.float()

        policy_loss = self.value_network(states, self.policy_network(states))
        policy_loss = -policy_loss.mean()

        # Compute bellman target value
        with torch.no_grad():
            next_action = self.target_policy_network(states_next)
            target_value = self.target_value_network(states_next, next_action)
            expected_value = rewards + (1.0 - dones) * self.gamma * target_value

        value = self.value_network(states, actions)
        value_loss = value_criterion(value, expected_value)

        # Compute step
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update targets
        for target_param, param in zip(self.target_value_network.parameters(), self.value_network.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.polyak) + param.data * self.polyak)

        for target_param, param in zip(self.target_policy_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.polyak) + param.data * self.polyak)

        return policy_loss.item(), value_loss.item()

    def save(self):
        torch.save(self.value_network.state_dict(), 'value_network.pth')
        torch.save(self.policy_network.state_dict(), 'policy_network.pth')

    @torch.no_grad()
    def predict(self, state):
        action = self.policy_network(state).cpu().numpy()
        return action
