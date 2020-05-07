#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Based on "Information-Directed Exploration for Deep Reinforcement Learning"
"""

import random
import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import rl_baselines
from rl_baselines.common import ReplayBuffer
from rl_baselines.common.logger import Logger
from rl_baselines.common.utils import set_global_seed

from rl_baselines.common.utils import quantile_huber_loss

class IDE():
    def __init__(
        self,
        env,
        network,
        n_quantiles=50,
        kappa=1,
        lamda=0.1,
        replay_start_size=50000,
        replay_buffer_size=1000000,
        gamma=0.99,
        update_target_frequency=10000,
        epsilon_12=0.00001,
        minibatch_size=32,
        learning_rate=1e-4,
        update_frequency=1,
        prior=0.01,
        adam_epsilon=1e-8,
        logging=False,
        log_folder=None,
        seed=None
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.replay_start_size = replay_start_size
        self.replay_buffer_size = replay_buffer_size
        self.gamma = gamma
        self.epsilon_12 = epsilon_12
        self.lamda = lamda
        self.update_target_frequency = update_target_frequency
        self.minibatch_size = minibatch_size
        self.learning_rate = learning_rate
        self.update_frequency = update_frequency
        self.adam_epsilon = adam_epsilon
        self.logging = logging
        self.logger = []
        self.timestep=0
        self.log_folder = log_folder

        self.env = env
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.seed = random.randint(0, 1e6) if seed is None else seed
        set_global_seed(self.seed, self.env)

        self.n_quantiles = n_quantiles

        self.network = network(self.env.observation_space, self.env.action_space.n*self.n_quantiles, self.env.action_space.n*self.n_quantiles, self.env.action_space.n).to(self.device)
        self.target_network = network(self.env.observation_space, self.env.action_space.n*self.n_quantiles, self.env.action_space.n*self.n_quantiles, self.env.action_space.n).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, eps=self.adam_epsilon)

        self.anchor1 = [p.data.clone() for p in list(self.network.output_1.parameters())]
        self.anchor2 = [p.data.clone() for p in list(self.network.output_2.parameters())]

        self.loss = quantile_huber_loss
        self.kappa = kappa
        self.prior = prior

    def learn(self, timesteps, verbose=False):

        if self.logging:
            self.logger = Logger(self.log_folder)

        # On initialise l'état
        state = torch.as_tensor(self.env.reset())
        score = 0
        t1 = time.time()

        for timestep in range(timesteps):

            is_training_ready = timestep >= self.replay_start_size

            # On prend une action
            action = self.act(state.to(self.device).float(), directed_exploration=True)

            # On execute l'action dans l'environnement
            state_next, reward, done, _ = self.env.step(action)

            # On stock la transition dans le replay buffer
            action = torch.as_tensor([action], dtype=torch.long)
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
                    self.logger.add_scalar('Episode_score', score, timestep)
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
                    loss = self.train_step(transitions)
                    if self.logging:
                        self.logger.add_scalar('Loss', loss, timestep)

                # Si c'est le moment, on update le target Q network on copiant les poids du network
                if timestep % self.update_target_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())

            if (timestep+1) % 250000 == 0:
                self.save(timestep=timestep+1)

        if self.logging:
            self.logger.save()
        
    def train_step(self, transitions):
        # huber = torch.nn.SmoothL1Loss()
        states, actions, rewards, states_next, dones = transitions

        # Calcul de la Q value via le target network (celle qui maximise)
        #print(self.target_network(states_next.float()).view(self.minibatch_size,self.env.action_space.n,self.n_quantiles).shape)
        with torch.no_grad():
            
            target1,target2,uncertainty_output = self.target_network(states_next.float())
            target1 = target1.view(self.minibatch_size,self.env.action_space.n,self.n_quantiles)
            target2 = target2.view(self.minibatch_size,self.env.action_space.n,self.n_quantiles)

            #Used to determine what action the current policy would have chosen in the next state
            target1_onpolicy,target2_onpolicy,uncertainty_output_onpolicy = self.network(states_next.float())
            target1_onpolicy = target1_onpolicy.view(self.minibatch_size,self.env.action_space.n,self.n_quantiles)
            target2_onpolicy = target2_onpolicy.view(self.minibatch_size,self.env.action_space.n,self.n_quantiles)

        best_action_idx = torch.mean((target1+target2)/2,dim=2).max(1, True)[1].unsqueeze(2)
        target1_gathered = target1.gather(1, best_action_idx.repeat(1,1,self.n_quantiles))
        target2_gathered = target2.gather(1, best_action_idx.repeat(1,1,self.n_quantiles))
        
        #uncertainty_target = uncertainty_output.gather(1,best_action_idx.squeeze(2))
        q_value_target = 0.5*target1_gathered\
            + 0.5*target2_gathered
        
        #Use current policy to determine what action would have been chosen in next state, using information directed sampling
        #Start with calculating regret
        action_means = torch.mean((target1_onpolicy+target2_onpolicy)/2,dim=2)
        delta_regret = torch.max(action_means + self.lamda*uncertainty_output_onpolicy,dim=1,keepdim=True)[0] \
            - (action_means - self.lamda*uncertainty_output_onpolicy)

        #Calculate aleatoric uncertainty (variance of quantiles - local epistemic uncertainty) and normalize
        uncertainty_local = torch.mean((target1_onpolicy-target2_onpolicy)**2,dim=2)/2
        aleatoric = torch.abs((torch.var(target1_onpolicy,dim=2)+torch.var(target2_onpolicy,dim=2))/2-uncertainty_local)
        uncertainties_aleatoric = aleatoric/(self.epsilon_12+torch.mean(aleatoric,dim=1,keepdim=True))

        #Calculate regret to information ratio and select actions
        information = torch.log(1 + uncertainty_output_onpolicy/uncertainties_aleatoric) + self.epsilon_12
        regret_info_ratio = delta_regret**2/information
        best_actions = regret_info_ratio.argmin(1,keepdim=True)

        #Use target network and selected actions as the target
        uncertainty_target = uncertainty_output.gather(1,best_actions)

        # Calcul de la TD Target
        td_target = rewards.unsqueeze(2).repeat(1,1,self.n_quantiles) \
            + (1 - dones.unsqueeze(2).repeat(1,1,self.n_quantiles)) * self.gamma * q_value_target

        # Calcul de la Q value en fonction de l'action jouée
        out1,out2,uncertainty = self.network(states.float())
        out1 = out1.view(self.minibatch_size,self.env.action_space.n,self.n_quantiles) 
        out2 = out2.view(self.minibatch_size,self.env.action_space.n,self.n_quantiles) 

        q_value1 = out1.gather(1, actions.unsqueeze(2).repeat(1,1,self.n_quantiles))
        q_value2 = out2.gather(1, actions.unsqueeze(2).repeat(1,1,self.n_quantiles))
        uncertainty = uncertainty.gather(1,actions)

        #Calculate local uncertainty with our method, and calculate TD target as per "Uncertainty Bellman equation" paper
        uncertainty_local = torch.mean((q_value1.detach()-q_value2.detach())**2,dim=2)/2
        uncertainty_TDtarget = uncertainty_local + (1-dones)*self.gamma**2*uncertainty_target

        #Calculate quantile losses
        loss1 = self.loss(q_value1.squeeze(), td_target.squeeze(), self.device, kappa=self.kappa)
        loss2 = self.loss(q_value2.squeeze(), td_target.squeeze(), self.device, kappa=self.kappa)

        #TD error for the uncertainty
        uncertainty_loss = F.mse_loss(uncertainty,uncertainty_TDtarget,reduction='mean')

        quantile_loss = loss1+loss2

        diff1=[]
        for i, p in enumerate(self.network.output_1.parameters()):
            diff1.append(torch.sum((p - self.anchor1[i])**2))

        diff2=[]
        for i, p in enumerate(self.network.output_2.parameters()):
            diff2.append(torch.sum((p - self.anchor2[i])**2))

        diff1 = torch.stack(diff1).sum()
        diff2 = torch.stack(diff2).sum()

        anchor_loss = self.prior*(diff1+diff2)

        loss = quantile_loss + anchor_loss + uncertainty_loss
        #print(anchor_loss/loss)

        # Update des poids
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def act(self, state, directed_exploration=False):

        action = self.predict(state,directed_exploration=directed_exploration)
        
        return action

    @torch.no_grad()
    def predict(self, state, directed_exploration=False):
        if not directed_exploration: #Choose greedily
            net1,net2,_ = self.network(state)
            net1 = net1.view(self.env.action_space.n,self.n_quantiles)
            net2 = net2.view(self.env.action_space.n,self.n_quantiles)
            action_means = torch.mean((net1+net2)/2,dim=1)
            action = action_means.argmax().item()
        else:
            net1,net2,uncertainties_epistemic = self.network(state)
            net1 = net1.view(self.env.action_space.n,self.n_quantiles)
            net2 = net2.view(self.env.action_space.n,self.n_quantiles)

            #Calculate regret
            action_means = torch.mean((net1+net2)/2,dim=1)
            delta_regret = torch.max(action_means + self.lamda*uncertainties_epistemic) - (action_means - self.lamda*uncertainties_epistemic)
            
            #Calculate and normalize aleatoric uncertainties (variance of quantile - local epistemic)
            uncertainties_epistemic_local = torch.mean((net1-net2)**2,dim=1)/2
            aleatoric = torch.abs((torch.var(net1,dim=1)+torch.var(net2,dim=1))/2-uncertainties_epistemic_local)
            uncertainties_aleatoric = aleatoric/(self.epsilon_12+torch.mean(aleatoric))

            #Use learned uncertainty and aleatoric uncertainty to calculate regret to information ratio and select actions
            information = torch.log(1 + uncertainties_epistemic/uncertainties_aleatoric) + self.epsilon_12
            regret_info_ratio = delta_regret**2/information
            action = regret_info_ratio.argmin().item()

        return action

    def save(self,timestep=None):
        if timestep is not None:
            filename = 'network_' + str(timestep) + '.pth'
        else:
            filename = 'network.pth'

        if self.log_folder is not None:
            save_path = self.log_folder + '/' +filename
        else:
            save_path = filename

        torch.save(self.network.state_dict(), save_path)

    def load(self,path):
        self.network.load_state_dict(torch.load(path,map_location='cpu'))