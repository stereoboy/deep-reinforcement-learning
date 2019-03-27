import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, sigma=0.05): # 0.017 -> 0.07 for bigger noise
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
#        self.in_features = in_features
#        self.out_features = out_features
#        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.noise_sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma))
        self.register_buffer('epsilon_weight', torch.Tensor(out_features, in_features))
        if bias:
#            self.bias = Paramter(torch.Tensor(out_features))
            self.noise_sigma_bias = nn.Parameter(torch.full((out_features,), sigma))
            self.register_buffer('epsilon_bias', torch.Tensor(out_features))
#        else:
#            self.register_parameter('bias', None)
        self.reset_parameters()

#    def reset_parameters(self):
#        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#        if self.bias is not None:
#            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#            bound = 1 / math.sqrt(fan_in)
#            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
#        return F.linear(input, self.weight, self.bias)
        self.epsilon_weight.normal_()
        weight = self.weight + self.noise_sigma_weight*self.epsilon_weight.data
        if self.bias is not None:
            self.epsilon_bias.normal_()
            bias = self.bias + self.noise_sigma_bias*self.epsilon_bias.data
        return F.linear(input, weight, bias)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))

class NoisyActor(nn.Module):
    """NoisyActor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(NoisyActor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = NoisyLinear(fc1_units, fc2_units)
        self.fc3 = NoisyLinear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
