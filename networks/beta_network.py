import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BetaNetwork(nn.Module):
    """Beta Network
    """

    def __init__(self, state_size, action_size, seed):
        """initializes network

        Params
        ======
            state_size (int or tuple): state space size
            action_size (int): action space size
            seed (int): random seed
        """

        super(BetaNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.linear1 = nn.Linear(state_size, 128)
        self.linear2 = nn.Linear(128, 32)
        self.linear_alpha = nn.Linear(32, action_size)
        self.linear_beta = nn.Linear(32, action_size)

    def forward(self, state):
        """network forward pass

        Params
        ======
            state (array): state (input to network)
        """

        hidden_state = F.relu(self.linear1(state))
        hidden_state = F.relu(self.linear2(hidden_state))
        alpha = F.softplus(self.linear_alpha(hidden_state)) + 1e-3
        beta = F.softplus(self.linear_beta(hidden_state)) + 1e-3

        return alpha, beta

    def sample(self, alpha, beta):
        """sample from beta distribution using parameters

        Params
        ======
            alpha (array): alpha parameter of beta distribution
            beta (array): beta parameter of beta distribution
        """

        dist = torch.distributions.beta.Beta(alpha, beta)

        action = dist.sample()

        return action * 2 - 1, torch.mean(torch.exp(dist.log_prob(action)), dim=-1)

    def get_probability(self, state, action):
        """get probability of a selected action

        Params
        ======
            state (array): state (input to network)
            action (array): selected action
        """

        alpha, beta = self(state)

        dist = torch.distributions.beta.Beta(alpha, beta)

        action = (action + 1) / 2
        action[action == 0] = 1e-6
        action[action == 1] = 1 - 1e-6

        # print(alpha)
        # print(beta)
        # print(dist.log_prob(action)

        return torch.mean(torch.exp(dist.log_prob(action)), dim=-1), dist.mean * 2 - 1, dist.variance * 2

    def update_var(self):
        pass