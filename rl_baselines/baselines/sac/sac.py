###
# Implementation of Soft Actor Critic Algorithm https://arxiv.org/pdf/1801.01290.pdf
# @ Benoit
###

import time

import torch
import torch.nn as nn
import torch.optim as optim

from rl_baselines.common import ReplayBuffer
from rl_baselines.common.logger import Logger


class SAC:

    def __init__(self,
                 env,
                 policy_network,
                 value_network,
                 soft_q_network,
                 alpha=0.2,
                 replay_start_size=50000,
                 replay_buffer_size=1000000,
                 update_target_frequency=10,
                 gamma=0.99,
                 polyak=1e-2,
                 minibatch_size=32,
                 policy_learning_rate=1e-3,
                 value_learning_rate=1e-3,
                 soft_q_learning_rate=1e-3,
                 update_frequency=4,
                 adam_epsilon=1e-8,
                 logging=False):

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self._replay_start_size = replay_start_size
        self._replay_buffer_size = replay_buffer_size
        self._update_target_frequency = update_target_frequency
        self._gamma = gamma
        self._polyak = polyak
        self._minibatch_size = minibatch_size
        self._policy_learning_rate = policy_learning_rate
        self._value_learning_rate = value_learning_rate
        self._soft_q_learning_rate = soft_q_learning_rate
        self._update_frequency = update_frequency
        self._adam_epsilon = adam_epsilon
        self._logging = logging

        self._env = env
        self._replay_buffer = ReplayBuffer(self._replay_buffer_size)

        # Init networks and load state dict
        self._policy_network = policy_network(self._env.observation_space, self._env.action_space, 256).to(self._device)
        self._value_network = value_network(self._env.observation_space, 256).to(self._device)
        self._target_value_network = value_network(self._env.observation_space, 256).to(self._device)
        self._target_value_network.load_state_dict(self._value_network.state_dict())

        self._soft_q_network1 = soft_q_network(self._env.observation_space, self._env.action_space, 256).to(self._device)
        self._soft_q_network2 = soft_q_network(self._env.observation_space, self._env.action_space, 256).to(self._device)

        # Set and optimizer
        self._policy_optimizer = optim.Adam(self._policy_network.parameters(), lr=self._policy_learning_rate, eps=self._adam_epsilon)
        self._value_optimizer = optim.Adam(self._value_network.parameters(), lr=self._value_learning_rate, eps=self._adam_epsilon)
        self._soft_q_optimizer1 = optim.Adam(self._soft_q_network1.parameters(), lr=self._soft_q_learning_rate, eps=self._adam_epsilon)
        self._soft_q_optimizer2 = optim.Adam(self._soft_q_network2.parameters(), lr=self._soft_q_learning_rate, eps=self._adam_epsilon)

    def _update_target(self, network, target_network):

        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self._polyak) + param.data * self._polyak)

    def _act(self, state, is_training_ready):
        # The output of the policy network must be in cpu !!
        if is_training_ready:
            with torch.no_grad():
                action = self._policy_network.get_action(state.unsqueeze(0))
                return action

        else:
            return self._env.action_space.sample()

    def learn(self, timesteps, verbose=False):

        if self._logging:
            logger = Logger()

        # On initialise l'état
        state = torch.FloatTensor(self._env.reset())
        done = False
        score = 0
        ep_len = 0
        t1 = time.time()

        for timestep in range(timesteps):

            is_training_ready = (timestep >= self._replay_start_size)

            # On prend une action
            action = self._act(state.to(self._device), is_training_ready)

            # On execute l'action dans l'environnement
            state_next, reward, done, _ = self._env.step(action)

            # On stock la transition dans le replay buffer
            action = torch.FloatTensor(action)
            reward = torch.FloatTensor([reward])
            done = torch.as_tensor([done], dtype=torch.float)
            state_next = torch.FloatTensor(state_next)
            self._replay_buffer.add(state, action, reward, state_next, done)

            score += reward.item()
            ep_len += 1

            if done:
                # On réinitialise l'état
                if verbose:
                    print(
                        "Timestep : {}, score : {:0.2f}, Time : {:0.2f} s, Episode length : {}".format(timestep, score, time.time() - t1, 3), ep_len)
                if self._logging:
                    logger.add_scalar('Episode_score', score, timestep)
                state = torch.FloatTensor(self._env.reset())
                done = False
                score = 0
                ep_len = 0
                t1 = time.time()
            else:
                state = state_next

            if is_training_ready:

                # Update du réseau principal
                if timestep % self._update_frequency == 0:

                    # On sample un minibatche de transitions
                    transitions = self._replay_buffer.sample(self._minibatch_size, self._device)

                    # On s'entraine sur les transitions selectionnées
                    policy_loss, value_loss, soft_q_loss = self._train_step(transitions)
                    if self._logging:
                        logger.add_scalar('policy_loss', policy_loss, timestep)
                        logger.add_scalar('value_loss', value_loss, timestep)
                        logger.add_scalar('soft_q_loss', soft_q_loss, timestep)

        if self._logging:
            logger.save()

    def _train_step(self, transitions):
        criterion = nn.MSELoss()

        states, actions, rewards, states_next, dones = transitions

        predicted_q_value1 = self._soft_q_network1(states, actions)
        predicted_q_value2 = self._soft_q_network2(states, actions)
        predicted_value = self._value_network(states)
        new_action, log_prob, epsilon, mean, log_std = self._policy_network.evaluate(states)
        del epsilon, mean, log_std

        # Training Q Function
        target_value = self._target_value_network(states_next)
        target_q_value = rewards + (1 - dones) * self._gamma * target_value
        q_value_loss1 = criterion(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = criterion(predicted_q_value2, target_q_value.detach())

        self._soft_q_optimizer1.zero_grad()
        self._soft_q_optimizer2.zero_grad()

        q_value_loss1.backward()
        q_value_loss2.backward()

        self._soft_q_optimizer1.step()
        self._soft_q_optimizer2.step()

        # Training Value function
        min_q_value = torch.min(self._soft_q_network1(states, new_action), self._soft_q_network2(states, new_action))
        target_value_func = min_q_value - self.alpha * log_prob
        value_loss = criterion(predicted_value, target_value_func.detach())

        self._value_optimizer.zero_grad()
        value_loss.backward()
        self._value_optimizer.step()

        # Training policy function
        policy_loss = (self.alpha * log_prob - self._soft_q_network1(states, new_action)).mean()

        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        self._update_target(self._value_network, self._target_value_network)

        return policy_loss.item(), value_loss.item(), q_value_loss1.item()
