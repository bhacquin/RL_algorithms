import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import numpy as np

class GCN(nn.Module):
    def __init__(self, observation_space, n_outputs, n_hid = 100):
        super(GCN, self).__init__()

        if len(observation_space.shape) != 2:
            raise NotImplementedError

        self.gc1 = GraphConvolution(2, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_hid)
        self.gc3 = GraphConvolution(n_hid, 1)

    def forward(self, obs):
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        x = obs[:,0:2,:]
        adj = obs[:,2:,:]
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        out = self.gc3(x, adj).squeeze()
        return out

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        adj_hat = adj + torch.eye(x.shape[-1])
        #D = -torch.FloatTensor(torch.sum(adj_hat, dim=1))
        #D = torch.diag_embed(D,dim1=1,dim2=2)
        #adj_hat = torch.matmul(torch.inverse(D),adj_hat)
        support = torch.matmul(x,adj_hat)
        output = torch.matmul(torch.transpose(self.weight,0,1),support)
        if self.bias is not None:
            output = output + self.bias.unsqueeze(-1)
            return output
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
