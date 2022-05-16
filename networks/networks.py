import torch
from importlib import reload
import networks.gaussian_network
reload(networks.gaussian_network)
from networks.gaussian_network import GaussianNetwork
import networks.beta_network
reload(networks.beta_network)
from networks.beta_network import BetaNetwork
import networks.dq_network
reload(networks.dq_network)
from networks.dq_network import DeepQNetwork
import networks.ddpg_actor_network
reload(networks.ddpg_actor_network)
from networks.ddpg_actor_network import DDPGActorNetwork
import networks.ddpg_critic_network
reload(networks.ddpg_critic_network)
from networks.ddpg_critic_network import DDPGCriticNetwork


def get_network(name, state_size, action_size, seed):
    """returns the initialized network loaded to the correct device
    
    Params
    ======
        name (str): name of network
        state_size (int or tuple): state space size
        action_size (int): action space size
        seed (int): random seed
    """
    
    if name == "gaussian":
        net = GaussianNetwork(state_size, action_size, seed)
    elif name == "beta":
        net = BetaNetwork(state_size, action_size, seed)
    elif name == "dqn":
        net = DeepQNetwork(state_size, action_size, seed)
    elif name == "ddpg_actor":
        net = DDPGActorNetwork(state_size, action_size, seed)
    elif name == "ddpg_critic":
        net = DDPGCriticNetwork(state_size, action_size, seed)
    else:
        raise NotImplementedError(f"Network ({name}) not found!")
        
    if torch.cuda.is_available():
        net = net.to("cuda")
        
    return net