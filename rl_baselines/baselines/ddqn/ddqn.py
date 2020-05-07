#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ted on Thu Jan  3 2019

@author: maxime
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_baselines.baselines import DQN


class DDQN(DQN):

    def _train_step(self, transitions):
        # huber = torch.nn.SmoothL1Loss()
        states, actions, rewards, states_next, dones = transitions
        states = states.float()
        states_next = states_next.float()

        with torch.no_grad():
            # Calcul de l'action via le network (celle qui maximise)
            actions_target = self.network(states_next).max(1, True)[1]

            # Calcul de la Q value via le target network (ce réseau évalue l'action choisie par network)
            q_value_target = self.target_network(states_next).gather(1, actions_target)

        # Calcul de la TD Target
        td_target = rewards + (1 - dones) * self._gamma * q_value_target

        # Calcul de la Q value en fonction de l'action jouée
        q_value = self.network(states).gather(1, actions)
        # Calcul de l'erreur
        loss = F.mse_loss(q_value, td_target, reduction='mean')

        # Update des poids
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

class DuelingMLP(nn.Module):
    def __init__(self, observation_space, n_outputs, hiddens=[100, 100]):
        super().__init__()

        if len(observation_space.shape) != 1:
            raise NotImplementedError
        else:
            n_inputs = observation_space.shape[0]
        
        layers = []
        for hidden in hiddens:
            layers.append(nn.Linear(n_inputs, hidden))
            layers.append(nn.ReLU())
            n_inputs = hidden
        self.hidden_layers = nn.Sequential(*layers)

        # note that n_inputs is now the size of the latent representation
        # i.e. the size of the output of hidden_layers
        self.value_estimator = nn.Linear(n_inputs, 1)
        self.advantages_estimator = nn.Linear(n_inputs, n_outputs)
        
    def forward(self, obs):
        H = self.hidden_layers(obs)
        value = self.value_estimator(H)
        advantages = self.advantages_estimator(H)
        return value + advantages - advantages.mean(-1, keepdim=True)  # advantages are centered
