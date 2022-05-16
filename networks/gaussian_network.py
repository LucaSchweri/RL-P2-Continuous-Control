import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def normal_pdf(val, mean, var):
    """pdf of normal distribution

    Params
    ======
        val (array): actual value
        mean (array): mean of normal distribution
        var (array): variance of normal distribution
    """

    a2 = torch.pow(val - mean, 2) / (2 * var)
    a = torch.exp(-1 * a2)

    b = 1 / torch.sqrt(2 * var * np.pi)
    return a * b


class GaussianNetwork(nn.Module):
    """Gaussian Network
    """

    def __init__(self, state_size, action_size, seed):
        """initializes network

        Params
        ======
            state_size (int or tuple): state space size
            action_size (int): action space size
            seed (int): random seed
        """

        super(GaussianNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.linear1 = nn.Linear(state_size, 128)
        self.linear2 = nn.Linear(128, 32)
        self.linear_mean = nn.Linear(32, action_size)

        self.var = 1.0

    def forward(self, state):
        """network forward pass

        Params
        ======
            state (array): state (input to network)
        """

        hidden_state = F.relu(self.linear1(state))
        hidden_state = F.relu(self.linear2(hidden_state))
        mean = torch.tanh(self.linear_mean(hidden_state))

        return mean, torch.full_like(mean, self.var)

    def sample(self, mean, var):
        """sample from normal distribution using parameters

        Params
        ======
            mean (array): mean of normal distribution
            var (array): variance of normal distribution
        """

        dist = torch.distributions.normal.Normal(mean, torch.sqrt(var))

        sample = dist.sample()

        return torch.clamp(sample, min=-1, max=1), torch.mean(normal_pdf(sample, mean, var), dim=-1)

    def get_probability(self, state, action):
        """get probability of a selected action

        Params
        ======
            state (array): state (input to network)
            action (array): selected action
        """

        mean, var = self(state)

        prob = normal_pdf(action, mean, var)

        return torch.mean(prob, dim=-1), mean, var

    def update_var(self):
        """updates variance

        Params
        ======
        """

        self.var *= 0.995

