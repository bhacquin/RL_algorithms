import copy
import torch
import gc
import time
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.modules.loss import CrossEntropyLoss
from sklearn.preprocessing import StandardScaler
from torch.distributions import Categorical
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset,DataLoader, RandomSampler
from torch.nn import MSELoss
import random
from multi_env import SubProcessEnv , MultiEnv
from rl_baselines.common.logger import Logger
from itertools import chain
# from .multiprocessing_env import SubprocVecEnv
from torch.distributions import MultivariateNormal
from sklearn.preprocessing import StandardScaler
from rl_baselines.common.logger import Logger
from rl_baselines.common.utils import set_global_seed
from torch import Tensor
from torch.distributions import Distribution
from gym import Space
import gym.spaces as spaces
from typing import Tuple
from typing import List, Union
from numba import jit
from torch.distributions import Categorical, Distribution, Normal
from torch.utils.data import Dataset, DataLoader
import math
####################################################################################################################
# CNN

def init_weights(m, gain):
    if (type(m) == nn.Linear) | (type(m) == nn.Conv2d):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.zeros_(m.bias)

#TO DO: import CNN deepmind
class CNN_Model(nn.Module):
    def __init__(self,input_dim, output_dim,value_dim):
        super(CNN_Model, self).__init__()
        # action mean range -1 to 1

        self.conv = nn.Sequential(nn.Conv2d(6, 32, 8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, stride=1),
                                  nn.ReLU())

        self.value = nn.Sequential(nn.Linear(4096, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, value_dim))

        self.policy = nn.Sequential(nn.Linear(4096, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, output_dim))

        self.conv.apply(lambda x: init_weights(x, np.sqrt(1)))
        self.value.apply(lambda x: init_weights(x, np.sqrt(1)))
        self.policy.apply(lambda x: init_weights(x, np.sqrt(1)))

        # self.action_var = torch.full((action_dim,), action_std * action_std).to(device)


    def forward(self, obs):
        if len(obs.shape) != 4:
            obs = obs.unsqueeze(0)

        obs = obs.permute(0, 3, 1, 2)
        obs = obs / 255
        obs = self.conv(obs)
        obs = obs.view(obs.size(0), -1)
        raw_value = self.value(obs)
        raw_policy = self.policy(obs)

        return  raw_policy,raw_value.tanh()



###################################################################################################################
# Dataset Custom
class NonSequentialDataset(Dataset):
    """
     * ``N`` - number of parallel environments
     * ``T`` - number of time steps explored in environments
    Dataset that flattens ``N*T*...`` arrays into ``B*...`` (where ``B`` is equal to ``N*T``) and returns such rows
    one by one. So basically we loose information about sequence order and we return
    for example one state, action and reward per row.
    It can be used for ``Model``'s that does not need to keep the order of events like MLP models.
    For ``LSTM`` use another implementation that will slice the dataset differently
    """

    def __init__(self, *arrays: np.ndarray) -> None:
        """
        :param arrays: arrays to be flattened from ``N*T*...`` to ``B*...`` and returned in each call to get item
        """
        super().__init__()
        self.arrays = [array.reshape(-1, *array.shape[2:]) for array in arrays]

    def __getitem__(self, index):
        return [array[index] for array in self.arrays]

    def __len__(self):
        return len(self.arrays[0])



def dataset_function(*arrays: np.ndarray) -> Dataset:
    return NonSequentialDataset(*arrays)

    
#####################################################################################################################
# Compute rewards and advantage
#@jit(nopython=True, nogil=True)
def discount(rewards: np.ndarray, estimate_of_last: np.ndarray, dones: np.ndarray, discount: float):
    r"""
    Calculates discounted reward according to equation:
    .. math:: G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V_{t+n-1}(S_{t+n})
    This function cares about episodes ends, so that if one row of the ``rewards`` matrix contains multiple episodes
    it will use information from ``dones`` to determine episode horizon.
    If the ``rewards`` array contains unfinished episode this function will use values from ``estimate_of_last`` to
    calculate the :math:`\gamma^n V_{t+n-1}(S_{t+n})` term
    *Note:* This function does not support n-step discounts calculation. For this functionality
            look at the Reward`/`Advantage` classes
    Legend for dimensions:
     * ``N`` - number of parallel agents
     * ``T`` - number of time steps
    :param rewards: array of shape ``N*T`` containing rewards for each time step
    :param estimate_of_last: array of shape ``(N,)`` containing value estimates for last value(:math:`V_{t+n-1}`)
    :param dones:  array of shape ``N*1`` containing information about episode ends
    :param discount: discount value(gamma)
    :return: array of shape ``N*T`` with discounted values for each step
    """

    v: np.ndarray = estimate_of_last
    ret = np.zeros_like(rewards)
    for timestep in range(rewards.shape[1] - 1, -1, -1):
        r, done = rewards[:, timestep], dones[:, timestep]
        v = (r + discount * v * (1. - done)).astype(ret.dtype)
        ret[:, timestep] = v
    return ret

class GeneralizedAdvantageEstimation:
    r"""
    Implementation of Generalized Advantage Estimator given by the equation:
    .. math:: \hat{A}_t = \delta_t + \gamma\lambda\delta_{t+1} + \cdots + (\gamma\lambda)^{T-t+1}\delta_{T-1}
    where
    .. math:: \delta_t = r_t + \gamma V(s_{t+1})-V(s_t)
    """

    def __init__(self, gamma, tau):
        """
        :param gamma: essentially it's the discount factor as we know it from n-step rewards
        :param lam: can be interpreted as the `n` in n-step rewards. Specifically setting it to 0 reduces the equation
               to be single step TD error, while setting it to 1 means there is no horizon so estimate over all steps
        """
        self.gamma = gamma
        self.tau = tau

    def discounted(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray) -> np.ndarray:
        td_errors = rewards + self.gamma * values[:, 1:] * (1. - dones) - values[:, :-1]
        return discount(td_errors, np.zeros_like(values[:, 0]), dones, self.tau * self.gamma)


class GeneralizedRewardEstimation:
    r"""
    Implementation of Generalized Advantage Estimator given by the equation:
    .. math:: \hat{A}_t = \delta_t + \gamma\lambda\delta_{t+1} + \cdots + (\gamma\lambda)^{T-t+1}\delta_{T-1}
    where
    .. math:: \delta_t = r_t + \gamma V(s_{t+1})-V(s_t)
    """

    def __init__(self, gamma, tau):
        """
        :param gamma: essentially it's the discount factor as we know it from n-step rewards
        :param lam: can be interpreted as the `n` in n-step rewards. Specifically setting it to 0 reduces the equation
               to be single step TD error, while setting it to 1 means there is no horizon so estimate over all steps
        """

        self.gamma = gamma
        self.tau = tau

    def discounted(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray) -> np.ndarray:
        td_errors = rewards + self.gamma * values[:, 1:] * (1. - dones) - values[:, :-1]
        return discount(td_errors, np.zeros_like(values[:, 0]), dones, self.tau * self.gamma) + values[:, :-1]

######################################################################################################################
# Rewards' Normalizer

class NoNormalizer:
    """
    Does no normalization on the array. Handy for observation spaces like ``gym.Discrete``
    """
    def partial_fit(self, array: np.ndarray) -> None:
        pass

    def transform(self, array: np.ndarray) -> np.ndarray:
        return array
    def partial_fit_transform(self, array: np.ndarray) -> np.ndarray:
        """
        Handy method to run ``partial_fit`` and ``transform`` at once
        :param array: array of shape ``N*T`` or ``N*T*any`` to be normalized
        :return: normalized array of same shape as input
        """
        self.partial_fit(array)
        return self.transform(array)

class StandardNormalizer:
    """
    Normalizes the input by subtracting the mean and dividing by standard deviation.
    Uses ``sklearn.preprocessing.StandardScaler`` under the hood.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def partial_fit(self, array: np.ndarray) -> None:
        self.scaler.partial_fit(self._reshape_for_scaler(array))

    def transform(self, array: np.ndarray) -> np.ndarray:
        return self.scaler.transform(self._reshape_for_scaler(array)).reshape(array.shape)

    @staticmethod
    def _reshape_for_scaler(array: np.ndarray):
        new_shape = (-1, *array.shape[2:]) if array.ndim > 2 else (-1, 1)
        return array.reshape(new_shape)

    def partial_fit_transform(self, array: np.ndarray) -> np.ndarray:
        """
        Handy method to run ``partial_fit`` and ``transform`` at once
        :param array: array of shape ``N*T`` or ``N*T*any`` to be normalized
        :return: normalized array of same shape as input
        """
        self.partial_fit(array)
        return self.transform(array)

######################################################################################################################
# Converters : from Env.space to distribution over actions
def for_space(space: Space):
    if isinstance(space, spaces.Discrete):
        print('Discret')
        return DiscreteConverter(space)

    elif isinstance(space, spaces.Box):
        print('Box')
        return BoxConverter(space)

class DiscreteConverter:
    """
    Utility class to handle ``gym.spaces.Discrete`` observation/action space
    """
    def __init__(self, space: spaces.Discrete) -> None:

        self.space = space
        self.loss = CrossEntropyLoss()

    @property
    def discrete(self) -> bool:
        return True

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.space.n,

    def distribution(self, logits: Tensor) -> Distribution:
        return Categorical(logits=logits)

    def reshape_as_input(self, array: np.ndarray):
        return  array.reshape(array.shape[0] * array.shape[1], -1)

    def action(self, tensor: Tensor) -> Tensor:
        return self.distribution(tensor).sample()

    def distance(self, policy_logits: Tensor, y: Tensor) -> Tensor:
        return self.loss(policy_logits, y.long())

    def state_normalizer(self):
        return NoNormalizer()

    def policy_out_model(self, in_features: int) -> nn.Module:
        return nn.Linear(in_features, self.shape[0])

class BoxConverter:
    """
    Utility class to handle ``gym.spaces.Box`` observation/action space
    """

    def __init__(self, space: spaces.Box) -> None:

        self.space = space
        self.loss = MSELoss()

    @property
    def discrete(self) -> bool:
        return False

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.space.shape

    def distribution(self, logits: Tensor) -> Distribution:
        assert logits.size(1) % 2 == 0
        mid = logits.size(1) // 2
        loc = logits[:, :mid]
        scale = logits[:, mid:]
        scale = scale.expand_as(loc)
        scale = torch.diag_embed(scale)
        # scale = torch.diag(scale.view(-1))
        # print('loc',loc.size())
        # print('scale', scale.size())
        return MultivariateNormal(loc,scale)
        # return Normal(loc, scale)

    def reshape_as_input(self, array: np.ndarray):
        return  array.reshape(array.shape[0] * array.shape[1], *array.shape[2:])

    def action(self, tensor: Tensor) -> Tensor:
        min = torch.tensor(self.space.low, device=tensor.device)
        max = torch.tensor(self.space.high, device=tensor.device)
        #return torch.max(torch.min(self.distribution(logits=tensor).sample(), max), min)
        return self.distribution(logits=tensor).sample()

    def distance(self, policy_logits: Tensor, y: Tensor) -> Tensor:
        return self.loss(self.action(policy_logits), y)

    def state_normalizer(self) :
        return StandardNormalizer()

    def policy_out_model(self, in_features: int) -> nn.Module:
        return NormalDistributionModule(in_features, self.shape[0])


class NormalDistributionModule(nn.Module):
    def __init__(self, in_features: int, n_action_values: int):
        super().__init__()
        self.policy_mean = nn.Linear(in_features, n_action_values)
        self.policy_std = nn.Parameter(torch.zeros(1, n_action_values)-0.7)

    def forward(self, x):
        policy = self.policy_mean(x)
        policy_std = self.policy_std.expand_as(policy).exp()
        return torch.cat((policy, policy_std), dim=-1)
################################################################################################################
# Runner

class Runner:
    """
    Runs the simulation on the environments using specified agent for choosing the actions
    """

    def __init__(self, env: MultiEnv, agent: 'Agent'):
        """
        :param env: environment to be used for the simulation
        :param agent: agent to be used to act on the environment
        """
        self.env = env
        self.agent = agent

    def run(self, n_steps: int, render: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs the simulation for specified number of steps and aggregates all the observations made on the environment
        :param n_steps: number of steps to run the simulation for
        :param render: whether to render the environment during simulation or not
        :return: returns tuple of(``T`` stands for time steps and usually is equal to ``n_steps``,
                 ``B`` number of environments in ``MultiEnv``):
                 * ``states`` - of shape N*T*(state space shape) Note ``T`` her is equal to ``n_steps + 1``.
                   This shape allows us to construct ``prev_states`` and ``next_states`` easily later by doing:
                   .. code-block:: python
                    prev_states = states[:-1]
                    next_states = states[1:]
                 * ``actions`` - of shape N*T*(action space shape)
                 * ``rewards`` - of shape N*T
                 * ``dones`` - of shape N*T
        """
        state = self.env.reset()
        states = np.empty(self.get_mini_batch_shape(state.shape, n_steps + 1),
                          dtype=self.env.dtype)  # +1 for initial state
        rewards = np.empty(self.get_mini_batch_shape((self.env.n_envs,), n_steps), dtype=self.env.dtype)
        dones = np.empty(self.get_mini_batch_shape((self.env.n_envs,), n_steps), dtype=self.env.dtype)
        actions = None
        states[:, 0] = state
        for step in range(n_steps):
            if render:
                self.env.render()

            action = self.agent.act(state)

            if step == 0:  # lazy init when we know the action space shape
                actions = np.empty(self.get_mini_batch_shape(action.shape, n_steps), dtype=self.env.dtype)

            state, reward, done, _ = self.env.step(action)

            states[:, step + 1] = state
            actions[:, step] = action
            rewards[:, step] = reward
            dones[:, step] = done
        return states, actions, rewards, dones

    def get_mini_batch_shape(self, observation_shape, n_steps):
        return (self.env.n_envs, n_steps, *observation_shape[1:])

class RandomRunner(Runner):
    def __init__(self, env: MultiEnv):
        super().__init__(env, RandomAgent(env.action_space))
#################################################################################################################
# Agents

class RandomAgent:
    """
    Just a random agent. It acts randomly by sampling action space
    """

    # noinspection PyMissingConstructor
    def __init__(self, action_space: Space):
        self.action_space = action_space

    def act(self, state) -> np.ndarray:
        return np.asarray([self.action_space.sample() for _ in state])

    def _train(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray):
        pass

class PPO:

    def __init__(self,
                 env,
                 network  = CNN_Model,
                 wrapper =None,
                 gamma=0.99,
                 tau=0.95,
                 v_clip_range = 0.5,
                 coef_entropy = 0.01,
                 coef_value = 0.5,
                 seed=1111,#random.randint(0,1e6),
                 num_epoch = 2,
                 verbose=True,
                 learning_rate=0.00025,
                 logging=False,
                 log_folder=None,
                 render= True,
                 horizon = 128,
                 epsilon = 0.1,  # clip ratio loss
                 scheduler = True,
                 n_mini_batches=32,
                 num_envs = 16,
                 device = torch.device("cpu"),
                 normalize_state = True,
                 normalize_reward = True,
                 clip_grad_norm=1.):
        self.device = device
        self.num_envs = num_envs

        #### wrapper
        if wrapper is not None:
            self.env = wrapper(env)
        else:
            self.env = env
        self.eval_env = self.env
        self.coef_entropy = coef_entropy
        self.coef_value = coef_value
        self.epsilon = epsilon #clip range for ratio surrogate loss
        self.v_clip_range = v_clip_range #clip for the value loss
        self.env = env
        self.horizon = horizon
        self.gamma = gamma
        self.tau = tau
        self.seed = seed
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.logging = logging
        self.log_folder = log_folder
        self.render = render
        set_global_seed(self.seed,self.env)
        self.scheduler_bool = scheduler
        ### Converter of the env.space
        self.state_converter = for_space(self.env.observation_space)
        self.action_converter = for_space(self.env.action_space)
        self.input_dim = self.state_converter.shape[0]
        self.policy_outputs = self.action_converter.shape[0]
        self.value_output = 1


        # multi_env = [self.make_env() for i in range(self.num_envs)]
        # self.multi_env = SubprocVecEnv(multi_env)
        self.env = MultiEnv(env.spec.id, num_envs, wrapper=wrapper)

        ## Definition of the two networks
        self.output_model = 64
        # self.old_network = network(self.env.observation_space, self.policy_outputs, self.value_output, width=64).to(self.device)

        self.model = network(self.env.observation_space, self.output_model, self.value_output).to(self.device)

        self.policy_out = self.action_converter.policy_out_model(self.output_model)
        # self.current_network.load_state_dict(self.old_network.state_dict())
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        #
        self.reward = GeneralizedRewardEstimation(gamma=0.95, tau=self.tau)

        self.advantage = GeneralizedAdvantageEstimation(gamma=0.95, tau=self.tau)
        self.n_mini_batches = n_mini_batches
        self.n_optimization_epochs = num_epoch
        self.clip_grad_norm = clip_grad_norm
        self.optimizer = torch.optim.Adam(chain(self.model.parameters(),self.policy_out.parameters()), self.learning_rate)

        self.reward_normalizer = StandardNormalizer() if normalize_reward else NoNormalizer()
        self.state_normalizer = self.state_converter.state_normalizer() if normalize_state else NoNormalizer()
        self.normalize_state = normalize_state

    def _tensors_to_device(self, *tensors: torch.Tensor) -> List[torch.Tensor]:
        return [tensor.to(self.device, torch.float) for tensor in tensors]


    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.tensor(array, device=self.device, dtype=torch.float)

    def loss(self,distribution_old: Distribution, value_old: Tensor, distribution: Distribution,
                value: Tensor, action: Tensor, reward: Tensor, advantage: Tensor):

        value_old_clipped = value_old + (value - value_old).clamp(-self.v_clip_range, self.v_clip_range)
        v_old_loss_clipped = (reward - value_old_clipped).pow(2)
        v_loss = (reward - value).pow(2)
        value_loss = torch.min(v_old_loss_clipped, v_loss).mean()

        # Policy loss
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        advantage.detach_()

        log_prob = distribution.log_prob(action)
        log_prob_old = distribution_old.log_prob(action)
        ratio = (log_prob - log_prob_old).exp().view(-1) ## ??? why does it explode here ?????


        surrogate = advantage * ratio
        surrogate_clipped = advantage * ratio.clamp(1 - self.epsilon, 1 + self.epsilon)
        policy_loss = torch.min(surrogate, surrogate_clipped).mean()
        # Entropy
        entropy = distribution.entropy().mean()

        # Total loss
        losses = policy_loss + self.coef_entropy * entropy - self.coef_value * value_loss
        if self.verbose:
            print('log_prob ', log_prob_old.min(), log_prob_old.max())
            print(f'policy_loss: {policy_loss} ')
            print(f'entropy: {entropy} ')
            print(f'value_loss: {value_loss} ')
            print(f'Losses: {losses}')
        total_loss = -losses
        return total_loss

    @torch.no_grad()
    def act(self, state : np.ndarray, deterministic=False) :
        state = self.state_normalizer.transform(state[:, None, :])
        reshaped_states = self.state_converter.reshape_as_input(state)
        logits, _ = self.model(torch.tensor(reshaped_states, device=self.device, dtype=torch.float))
        logits = self.policy_out(logits)

        if deterministic:
            return logits.cpu().detach().numpy()
        return self.action_converter.action(logits).cpu().detach().numpy()


    def learn(self,epochs: int, n_steps: int, initialization_steps: int = 1000,  render: bool = False):
        """
        Trains the agent for ``epochs`` number of times by running simulation on the environment for ``n_steps``
        :param epochs: number of epochs of training
        :param n_steps: number of steps made in the environment each epoch
        :param initialization_steps: number of steps made on the environment to gather the states then used for
               initialization of the state normalizer
        :param render: whether to render the environment during learning
        """

        if initialization_steps and self.normalize_state:

            s, _, _, _ = RandomRunner(self.env).run(initialization_steps)
            self.state_normalizer.partial_fit(s)

        for epoch in tqdm(range(epochs)):

            states, actions, rewards, dones = Runner(self.env, self).run(n_steps, render)
            states = self.state_normalizer.partial_fit_transform(states)
            #rewards = self.curiosity.reward(rewards, states, actions)
            rewards = self.reward_normalizer.partial_fit_transform(rewards)
            self.train(states, actions, rewards, dones)
            print(f'Epoch: {epoch} done')
        self.eval(n_steps=20, render=True)




    def train(self,states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray):
        policy_old, values_old = self.model(self._to_tensor(self.state_converter.reshape_as_input(states)))
        policy_old = self.policy_out(policy_old)
        policy_old = policy_old.detach().view(*states.shape[:2], -1)
        values_old = values_old.detach().view(*states.shape[:2])
        values_old_numpy = values_old.cpu().detach().numpy()
        discounted_rewards = self.reward.discounted(rewards, values_old_numpy, dones)
        advantages = self.advantage.discounted(rewards, values_old_numpy, dones)
        dataset = dataset_function(policy_old[:, :-1], values_old[:, :-1], states[:, :-1], states[:, 1:], actions,
                                     discounted_rewards, advantages)

        loader = DataLoader(dataset, batch_size=len(dataset) // self.n_mini_batches, shuffle=True)
        with torch.autograd.detect_anomaly():
            for _ in range(self.n_optimization_epochs):
                for tuple_of_batches in loader:
                    (batch_policy_old, batch_values_old, batch_states, batch_next_states,
                     batch_actions, batch_rewards, batch_advantages) = self._tensors_to_device(*tuple_of_batches)
                    batch_policy, batch_values = self.model(batch_states)
                    batch_policy = self.policy_out(batch_policy)
                    batch_values = batch_values.squeeze()
                    distribution_old = self.action_converter.distribution(batch_policy_old)
                    distribution = self.action_converter.distribution(batch_policy)

                    loss: Tensor = self.loss(distribution_old, batch_values_old, distribution, batch_values,
                                             batch_actions, batch_rewards, batch_advantages)
                    #loss = self.curiosity.loss(loss, batch_states, batch_next_states, batch_actions)
                    # print('loss:', loss)
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.optimizer.step()
                self.eval(n_steps=1, render=False)


    def eval(self, n_steps: int, render: bool = False):
        #Runner(self.env, self).run(n_steps, render)### TO DO


        score = []
        for step in range(n_steps):
            done = False
            rewards = 0
            state = self.eval_env.reset()

            while done == False:
                if render:
                    self.eval_env.render()

                tuple = [x for x in list(self.state_converter.shape)]
                tuple.insert(0,-1)#.insert(0, -1))
                tuple = (tuple)
                #action = self.act(np.array(state).reshape(tuple))
                action = self.act(np.expand_dims(np.array(state),0))#.reshape(-1,self.state_converter.shape))

                state, reward, done, _ = self.eval_env.step(action[0])

                rewards += reward
                done = done
            if self.verbose:
                print(rewards)
            score.append(rewards)
        print(np.mean(score))
        return score




