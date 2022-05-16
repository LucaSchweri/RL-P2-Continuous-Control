import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPGActorNetwork(nn.Module):
    """DDPG Actor Network
    """
    
    def __init__(self, state_size, action_size, seed):
        """initializes network
    
        Params
        ======
            state_size (int or tuple): state space size
            action_size (int): action space size
            seed (int): random seed
        """
        
        super(DDPGActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.linear1 = nn.Linear(state_size, 128)
        self.linear2 = nn.Linear(128, 32)
        self.linear3 = nn.Linear(32, action_size)

    def forward(self, state, add_noise=False):
        """network forward pass
    
        Params
        ======
            state (array): state (input to network)
        """
        
        hidden_state = F.relu(self.linear1(state))
        hidden_state = F.relu(self.linear2(hidden_state))
        best_action = self.linear3(hidden_state)

        if add_noise:
            best_action += torch.randn_like(best_action)

        best_action = torch.clamp(best_action, min=-1, max=1)
        
        return best_action