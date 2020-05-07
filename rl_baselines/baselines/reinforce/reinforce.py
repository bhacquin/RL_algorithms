"""
Implementation of the Vanilla policy gradient algorithm
@author : Benoit ROBAGLIA

NOT GPU optimized
"""

import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from rl_baselines.common.logger import Logger
from rl_baselines.common.utils import set_global_seed


class Reinforce:

    def __init__(self,
                 env,
                 network,
                 gamma=0.99,
                 seed=random.randint(0,1e6),
                 verbose=True,
                 learning_rate=0.001,
                 logging=True):

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.env = env
        self.gamma = gamma
        self.seed = seed
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.logging = logging

        set_global_seed(self.seed,self.env)

        self.network = network(self.env.observation_space, self.env.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)

    @torch.no_grad()
    def act(self, state):
        network_out = self.network(state)
        probs = torch.exp(F.log_softmax(network_out.squeeze(), dim=-1))
        dist = Categorical(probs)
        action = dist.sample().item()
        log_prob = torch.log(probs[action])
        return action, log_prob

    def calculate_returns(self, rewards):
        returns = torch.zeros(len(rewards))
        running_sum = 0
        for i in reversed(range(len(rewards))):
            running_sum = running_sum * self.gamma + rewards[i]
            returns[i] = running_sum

        return returns

    def learn(self, timesteps):

        if self.logging:
            logger = Logger()

        state = torch.tensor(self.env.reset(), dtype=torch.float, device=self.device)
        done = False

        states = []
        actions = []
        rewards = []
        score = 0
        episode_count = 0

        for timestep in range(timesteps):

            if not done:
                # On prend une action
                action, _ = self.act(state)

                # On execute l'action dans l'environnement
                next_state, reward, done, _ = self.env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)

                score += reward

                # On sauvegarde les transitions
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            if done:
                # On discount les rewards, normalise les state values et on calcule l'avantage
                returns = self.calculate_returns(rewards)

                # Update du réseau principal
                loss = self.train_step(states, actions, returns)

                state = torch.tensor(self.env.reset(), dtype=torch.float, device=self.device)

                if self.verbose:
                    print("Episode {} : score {}".format(episode_count, score))
                if self.logging:
                    logger.add_scalar('Episode_score', score, timestep)
                    logger.add_scalar('Loss', loss, timestep)

                done = False
                states = []
                actions = []
                rewards = []
                score = 0
                episode_count += 1

        if self.logging:
            logger.save()

    def train_step(self, states, actions, returns):
        states = torch.stack(states).to(self.device) # Tensor de taille (batch x dim space)
        actions = torch.tensor(actions, device=self.device)
        returns = returns.to(self.device)

        # On récupère les log probs
        policy_out = self.network(states)
        log_probs = F.log_softmax(policy_out, dim=-1)
        log_probs = log_probs.gather(1, actions.unsqueeze(1))

        self.optimizer.zero_grad()
        loss = torch.mean(-log_probs * returns)
        loss.backward()
        self.optimizer.step()

        return loss.item()
