import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, input_shape, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            input_shape (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        #self.seed = torch.manual_seed(seed)

        self.conv = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
                )

        conv_out_size = int(np.prod(self.conv(torch.zeros(1, *input_shape)).size()))

        self.fc = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, action_size)
                )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        batch_size = state.size()[0]
        fx = state.float()/256 - 0.5
        #fx = state - 0.5
#        print("===================================")
#        print(fx.shape)
        conv_out = self.conv(fx)
#        print(conv_out.shape)
        conv_out = self.conv(fx).view(batch_size, -1)
        return self.fc(conv_out)
