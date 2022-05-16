from importlib import reload
import agents.ppo_agent
reload(agents.ppo_agent)
from agents.ppo_agent import PPOAgent
import agents.a2c_agent
reload(agents.a2c_agent)
from agents.a2c_agent import A2CAgent
import agents.ddpg_agent
reload(agents.ddpg_agent)
from agents.ddpg_agent import DDPGAgent


def get_agent(name, agent_config, state_size, action_size, seed):
    """returns the initialized agent
    
    Params
    ======
        name (str): name of agent
        agent_config (dict): configurations for the specified agent
        state_size (int or tuple): state space size
        action_size (int): action space size
        seed (int): random seed
    """
    
    if name == "ppo":
        agent = PPOAgent(agent_config[name], state_size, action_size, seed)
    elif name == "a2c":
        agent = A2CAgent(agent_config[name], state_size, action_size, seed)
    elif name == "ddpg":
        agent = DDPGAgent(agent_config[name], state_size, action_size, seed)
    else:
        raise NotImplementedError(f"Agent ({name}) not found!")
        
    return agent