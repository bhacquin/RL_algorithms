import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m, gain):
    if (type(m) == nn.Linear):
        nn.init.xavier_normal_(m.weight, gain)


class PolicyNetworkSAC(nn.Module):

    def __init__(self, observation_space, action_space, hidden_size, log_std_min=-20, log_std_max=2):
        super(PolicyNetworkSAC, self).__init__()

        num_inputs = observation_space.shape[0]
        num_actions = action_space.shape[0]

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear_mean = nn.Linear(hidden_size, num_actions)
        self.linear_logstd = nn.Linear(hidden_size, num_actions)

        init_weights(self.linear1, 1)
        init_weights(self.linear2, 1)
        init_weights(self.linear_mean, 1)
        init_weights(self.linear_logstd, 1)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.linear_mean(x)
        logstd = self.linear_logstd(x)
        logstd = torch.clamp(logstd, self.log_std_min, self.log_std_max)
        return mean, logstd

    def evaluate(self, state, epsilon=1e-6):
        mean, logstd = self(state)
        std = logstd.exp()

        normal = torch.distributions.Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z)
        log_prob = torch.distributions.Normal(mean, std).log_prob(mean + std * z) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, logstd

    def get_action(self, state):
        mean, logstd = self(state)
        std = logstd.exp()

        normal = torch.distributions.Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z)

        action = action[0].detach().cpu().numpy()
        return action


class QValueNetwork(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size):
        super(QValueNetwork, self).__init__()

        num_inputs = observation_space.shape[0]
        num_actions = action_space.shape[0]

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        init_weights(self.linear1, 1)
        init_weights(self.linear2, 1)
        init_weights(self.linear3, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size=256):
        super(PolicyNetwork, self).__init__()

        num_inputs = observation_space.shape[0]
        num_actions = action_space.shape[0]

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        init_weights(self.linear1, 1)
        init_weights(self.linear2, 1)
        init_weights(self.linear3, 1)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x


class ValueNetwork(nn.Module):
    def __init__(self, observation_space, hidden_dim):
        super(ValueNetwork, self).__init__()

        num_inputs = observation_space.shape[0]

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        init_weights(self.linear1, 1)
        init_weights(self.linear2, 1)
        init_weights(self.linear3, 1)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x