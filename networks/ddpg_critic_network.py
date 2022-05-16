import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPGCriticNetwork(nn.Module):
    """DDPG Critic Network
    """
    
    def __init__(self, state_size, action_size, seed):
        """initializes network
    
        Params
        ======
            state_size (int or tuple): state space size
            action_size (int): action space size
            seed (int): random seed
        """
        
        super(DDPGCriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.linear1 = nn.Linear(state_size + action_size, 128)
        self.linear2 = nn.Linear(128, 32)
        self.linear3 = nn.Linear(32, 1)

    def forward(self, state):
        """network forward pass
    
        Params
        ======
            state (array): state (input to network)
        """
        
        hidden_state = F.relu(self.linear1(state))
        hidden_state = F.relu(self.linear2(hidden_state))
        state_action_value = self.linear3(hidden_state)
        
        return state_action_value