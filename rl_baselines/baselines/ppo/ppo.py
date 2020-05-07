"""
Implementation of the proximal policy gradient algorithm
@author : Bastien van Delft

NOT GPU optimized
"""
import copy
import torch
import gc
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset,DataLoader, RandomSampler
import random
#from tensorboardX import SummaryWriter
from rl_baselines.common.logger import Logger
from .multiprocessing_env import SubprocVecEnv
from torch.distributions import MultivariateNormal

from rl_baselines.common.logger import Logger
from rl_baselines.common.utils import set_global_seed

use_cuda = torch.cuda.is_available()




###


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


class DiagGaussian(nn.Module):
    def __init__(self, num_outputs):
        super(DiagGaussian, self).__init__()

        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = x

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)

        return torch.distributions.Normal(action_mean, action_logstd.exp()), action_logstd.exp()









class PPO:

    def __init__(self,
                 env,
                 network,
                 gamma=0.99,
                 tau=0.15,
                 coef_entropy = 0.01,
                 seed=random.randint(0,1e6),
                 verbose=True,
                 learning_rate=0.00025,
                 logging=False,
                 log_folder=None,
                 render= True,
                 horizon = 128,
                 epsilon = 0.1,
                 scheduler = True,
                 output2_dim=1,
                 num_envs = 16):

        self.num_envs = num_envs
        self.coef_entropy = coef_entropy
        self.output2_dim = output2_dim
        self.epsilon = epsilon
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.env = env
        self.horizon = horizon
        self.gamma = gamma
        self.action_std = 0.5

        self.tau = tau
        self.seed = seed
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.logging = logging
        self.log_folder = log_folder
        self.num_epoch = 4
        self.render = render
        set_global_seed(self.seed,self.env)
        self.scheduler_bool = scheduler
        # if self.env.action_space.__class__.__name__ == "Discrete":
        #     self.num_outputs = self.env.action_space.n
        #     self.dist = Categorical()
        #     print("Categorical")
        # elif self.env.action_space.__class__.__name__ == "Box":
        #     self.num_outputs = self.env.action_space.shape[0]
        #     self.dist = DiagGaussian(self.num_outputs)
        # elif self.env.action_space.__class__.__name__ == "MultiBinary":
        #     self.num_outputs = self.env.action_space.shape[0]
        #     self.dist = Bernoulli()
        if self.env.action_space.__class__.__name__ == "Discrete":
            self.num_outputs = self.env.action_space.n
        else :
            self.num_outputs = self.env.action_space.shape[0]
            self.dist = DiagGaussian(num_outputs = self.num_outputs)

        self.action_var = torch.full((self.num_outputs,), self.action_std * self.action_std)
        self.network = network(self.env.observation_space, self.num_outputs,self.output2_dim,width=64).to(self.device)
        self.current_network = network(self.env.observation_space, self.num_outputs,self.output2_dim,width=64).to(self.device)
        self.current_network.load_state_dict(self.network.state_dict())
        if self.env.action_space.__class__.__name__ != "Discrete":
            self.optimizer = optim.Adam([{'params' : self.current_network.parameters(),'lr' : self.learning_rate}])#,{'params' : self.dist.parameters(),'lr' : self.learning_rate}])
        else:
            self.optimizer = optim.Adam(self.current_network.parameters(), lr=self.learning_rate)
        self.envs = [self.make_env() for i in range(self.num_envs)]
        self.envs = SubprocVecEnv(self.envs)


    def make_env(self):
        def _thunk():
            # env = gym.make(env_name)
            # env = make_atari(self.env_id)
            # env = wrap_deepmind(env)
            env = copy.deepcopy(self.env)
            return env

        return _thunk


    @torch.no_grad()
    def act(self, state,deterministic = False,  test = False):
        network_out, value_out = self.network.forward(state)

        if self.env.action_space.__class__.__name__ == "Discrete":
            probs = torch.exp(F.log_softmax(network_out, dim=-1))
            dist = Categorical(probs)

            if deterministic:
                action = dist.mode().unsqueeze(1)
            else:
                action = dist.sample().unsqueeze(1)
            action_logprob = torch.log(probs.gather(1, action)).squeeze()

        elif self.env.action_space.__class__.__name__ == "Box":
            network_out, _ = self.network.forward(state)
            action_mean = torch.tanh(network_out)*2
            cov_mat = torch.diag(self.action_var)
            if self.num_outputs > 1:
                dist = MultivariateNormal(action_mean, cov_mat)
                action = dist.sample()
                action_logprob = dist.log_prob(action)
            else :

                dist, scale = self.dist(network_out)
                action = dist.sample()
                #print('std', scale)
                action_logprob = dist.log_prob(action)

        # log_prob = torch.log(probs[action])
        #if random.random()>0.97 and not test:
        #    print(probs)
        return action.squeeze().cpu(), action_logprob, value_out.squeeze()

    @torch.no_grad()
    def calculate_returns(self, rewards, values, next_states):
        returns = torch.zeros(len(rewards))
        running_sum = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) -1:
                with torch.no_grad():
                    _ , next_value = self.network.forward(torch.tensor(next_states[-1]))
                running_sum = next_value.item()
                returns[i] = running_sum * self.gamma + rewards[i]
            else:
                running_sum = running_sum * self.gamma + rewards[i]
                returns[i] = running_sum

        return returns

    @torch.no_grad()
    def compute_gae(self, next_value, rewards, dones, values, gamma=0.99, tau=0.95):


        values = values + [next_value.squeeze()]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1-dones[step]) - values[step]
            gae = delta + self.gamma * self.tau * (1-dones[step]) * gae
            returns.insert(0, gae + values[step])

        returns = torch.stack(returns)


        return returns

    def learn(self, timesteps):

        if self.logging:
            logger = Logger(self.log_folder)

        if self.scheduler_bool:
            self.scheduler = CosineAnnealingLR(self.optimizer, timesteps, eta_min=7e-5, last_epoch=-1)

        state = torch.tensor(self.envs.reset(), dtype=torch.float, device=self.device)
        done = False
        t1 = time.time()
        states = []

        actions = []
        rewards = []
        log_probs = []
        values = []
        dones = []
        score = 0
        episode_count = 0
        frames = 0

        for timestep in range(timesteps):
            state = torch.tensor(self.envs.reset(), dtype=torch.float, device=self.device)

            while frames < (self.horizon):

                #### Scheduler
                if self.scheduler_bool:
                    # self.scheduler.step()
                    self.learning_rate = 0.99999 * self.learning_rate
                    if self.env.action_space.__class__.__name__ != "Discrete":
                        self.optimizer = optim.Adam(
                            [{'params': self.current_network.parameters(), 'lr': self.learning_rate},
                             {'params': self.dist.parameters(), 'lr': self.learning_rate}])
                    else:
                        self.optimizer = optim.Adam(self.current_network.parameters(), lr=self.learning_rate)
                    # self.optimizer = optim.Adam(self.current_network.parameters(), lr=self.learning_rate)
                    self.epsilon = 0.99999 * self.epsilon
                    self.coef_entropy = 0.9999 * self.coef_entropy
                frames +=1
                # self.epsilon = 0.9995 * self.epsilon
                # self.coef_entropy = 0.9925 * self.coef_entropy
                # Rendering
                #if self.render == True:
                #    self.envs.render()

                # On prend une action
                #print('state init', state)
                action, log_prob, value = self.act(state)

                # On execute l'action dans l'environnement
                #print("action ", action.size(), type(action))
                action_made = action.data.numpy()
                if self.num_outputs == 1 and self.env.action_space.__class__.__name__ == "Box":
                    action_made = np.expand_dims(action_made, axis=-1)
                next_state, reward, done, _ = self.envs.step(action_made)

                next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)

                score += reward

                # On sauvegarde les transitions
                states.append(state)
                dones.append(torch.from_numpy(done.astype(float)).float().to(self.device))
                actions.append(action)
                rewards.append(torch.from_numpy(reward).float().to(self.device))
                log_probs.append(log_prob)
                values.append(value)



                if done.any():
                    state = torch.tensor(self.envs.reset(), dtype=torch.float, device=self.device)
                    if self.verbose:
                        #print("Episode {} : score {}".format(episode_count, score))
                        pass
                else :
                    state = next_state

            # On discount les rewards, normalise les state values et on calcule l'avantage

            _ , next_value = self.network.forward(next_state)
            returns = self.compute_gae(next_value, rewards, dones, values, gamma=0.99, tau=0.95)


            # Update du réseau principal
            loss = self.train_step(states, actions, returns, log_probs, values)
            if episode_count % 100 == 0:
                torch.save(self.current_network,'Pong_network.pt')

            if self.verbose :
                print("Timestep : {}, score : {}, Time : {} s".format(timestep, score, round(time.time() - t1, 3)))
                self.test(timestep)

            t1 = time.time()
            if self.logging:
                logger.add_scalar('Episode_score', score, timestep)
                logger.add_scalar('Loss', loss, timestep)
                #

            done = False
            states = []
            actions = []
            rewards = []
            log_probs = []
            values = []
            dones = []
            score = 0
            frames = 0
            episode_count += 1
            gc.collect()

        if self.logging:
            logger.save()




    def test(self,timestep=0, render = False):
        state = torch.tensor(self.env.reset(), dtype=torch.float, device=self.device)
        done = False

        score = 0

        with torch.no_grad():

            while not done:

                # Rendering
                if render == True:
                    self.env.render()

                # On prend une action
                action, log_prob, value = self.act(state.unsqueeze(0), test = True)
                action_made = action.data.numpy()
                if self.num_outputs == 1 and self.env.action_space.__class__.__name__ == "Box":
                    action_made = np.expand_dims(action_made, axis=-1)
                next_state, reward, done, _ = self.env.step(action_made)
                # On execute l'action dans l'environnement
                # next_state, reward, done, _ = self.env.step(action.item())
                next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
                state = next_state
                score += reward

            if self.verbose:
                print("Episode {} : score {}".format(1, score))



        return


    def train_step(self, states, actions, returns, log_probs, values):

        mse = torch.nn.MSELoss()
        states = torch.stack(states).cpu()#.to(self.device) # Tensor de taille (batch x dim space)

        view = []
        view.append(len(states)*self.num_envs)
        for i in range(2, len(states.size())):
            view.append(states.size(i))
        #states = states.view(-1, states.size(2), states.size(3), states.size(4))
        states = states.view(view)

        actions = torch.stack(actions).cpu()#.to(self.device)
        actions = actions.view(-1)

        returns = returns.detach().cpu()#to(self.device).detach()
        returns = returns.view(-1)

        log_probs_old = torch.stack(log_probs).cpu()#.to(self.device)
        log_probs_old = log_probs_old.view(-1)

        values_old = torch.stack(values).cpu()#.to(self.device).detach()
        values_old = values_old.view(-1)

        self.network.load_state_dict(self.current_network.state_dict())


        #### Procéder par Batch
        for _ in range(self.num_epoch):
            dataset = TensorDataset(states,actions.unsqueeze(1),returns.unsqueeze(1),log_probs_old.unsqueeze(1),values_old.unsqueeze(1))



            dataloader = DataLoader(dataset,batch_size=32,num_workers=0, drop_last=True, shuffle=True)


            for states_batch, actions_batch, returns_batch, log_probs_old_batch, values_old_batch in dataloader:
                states_batch = states_batch.to(self.device)
                actions_batch = actions_batch.to(self.device)
                returns_batch = returns_batch.to(self.device)
                log_probs_old_batch = log_probs_old_batch.to(self.device)
                values_old_batch = values_old_batch.to(self.device)

                # On récupère les log probs du nouveau réseau
                policy_out, state_values = self.current_network.forward(states_batch)
                state_values = state_values.squeeze(1)

                if self.env.action_space.__class__.__name__ == "Discrete":
                    #probs_full = F.softmax(policy_out, dim = -1)
                    #probs = probs_full.gather(1, actions_batch).squeeze()
                    log_probs_full = F.log_softmax(policy_out, dim=-1)
                    log_probs_new = log_probs_full.gather(1, actions_batch).squeeze()
                    entropy = torch.sum(log_probs_full.squeeze() * torch.exp(log_probs_full).squeeze(), dim=-1)
                elif self.env.action_space.__class__.__name__ == "Box":
                    action_mean = torch.tanh(torch.squeeze(policy_out))
                    cov_mat = self.action_var.expand_as(action_mean)
                    if self.num_outputs > 1:
                        dist = MultivariateNormal(action_mean, cov_mat)
                        log_probs_new = dist.log_prob(actions_batch)
                    else:
                        dist , scale =  self.dist(policy_out)
                        #print('scale', scale)

                        log_probs_new = dist.log_prob(actions_batch)

                        # print('entropy', dist.entropy())
                        # print('dist_scale', dist.scale)
                        # print(self.optimizer.param_groups[1])
                    entropy = dist.entropy().sum(-1)

                # On valorise les états de la trajectoire via notre nn de value, on normalise et on calcule l'avantage

                #state_values_detach = state_values.detach()  # retire le gradient du tenseur car on ne veux pas optimiser via l'avantage
                avantages = returns_batch.squeeze() - values_old_batch.squeeze()
                # avantages = (avantages - torch.mean(avantages))/(torch.std(avantages)+ 10e-8)

                self.optimizer.zero_grad()




                loss_policy_no_clipped = torch.exp(log_probs_new-log_probs_old_batch.squeeze()).squeeze() * avantages.squeeze()
                loss_policy_clipped = torch.clamp(torch.exp(log_probs_new-log_probs_old_batch.squeeze()), 1 - self.epsilon, 1 + self.epsilon).squeeze() * avantages

                surrogate_loss = -torch.min(loss_policy_no_clipped, loss_policy_clipped).mean()

                loss_value = mse(returns_batch.squeeze(), state_values)

                total_loss = surrogate_loss + 0.5 * loss_value + self.coef_entropy * entropy.mean()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.current_network.parameters(), 0.5)
                self.optimizer.step()
        try:
            print('entropy',entropy.mean())
            print('surrogate_loss', surrogate_loss)
            print('loss_value', loss_value)
            a = total_loss.item()

        except :
            print(states.size())
            return
            pass

        return total_loss.item()

    @torch.no_grad()
    def predict(self, state):

        action = self.network(state)[0].view(-1).argmax().item()
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

    def load(self, network = 'network.pth'):

        self.network = torch.load(network)
        self.current_network=torch.load(network)