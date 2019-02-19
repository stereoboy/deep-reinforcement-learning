import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, fc3_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DuelQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2_val = nn.Linear(fc1_units, fc2_units)
        self.fc2_adv = nn.Linear(fc1_units, fc2_units)
        self.fc3_val = nn.Linear(fc2_units, 1)
        self.fc3_adv = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x_val = F.relu(self.fc2_val(x))
        x_adv = F.relu(self.fc2_adv(x))
        val = self.fc3_val(x_val)
        adv = self.fc3_adv(x_adv)
        return val + adv - adv.mean()

class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, sigma=0.017):
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

class NoisyDuelQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, fc3_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(NoisyDuelQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2_val = NoisyLinear(fc1_units, fc2_units)
        self.fc2_adv = NoisyLinear(fc1_units, fc2_units)
        self.fc3_val = NoisyLinear(fc2_units, 1)
        self.fc3_adv = NoisyLinear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x_val = F.relu(self.fc2_val(x))
        x_adv = F.relu(self.fc2_adv(x))
        val = self.fc3_val(x_val)
        adv = self.fc3_adv(x_adv)
        return val + adv - adv.mean()
