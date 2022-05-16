import numpy as np

import torch
import torch.optim as optim

from importlib import reload
import networks.networks
reload(networks.networks)
from networks.networks import get_network

device = "cuda" if torch.cuda.is_available() else "cpu"

class PPOAgent():
    """PPO Agent
    
    Configurations (config.json)
    ======
        network (str): name of network
        lr (float): learning rate
        discount_factor (float): discount factor
        clip_epsilon (float): clip of the probability ratio in the surrogate function
        sgd_epochs (int): how many update steps per episode
    """

    def __init__(self, config, state_size, action_size, seed=0):
        """initializes agent
    
        Params
        ======
            config (dict): agent configurations
            state_size (int or tuple): state space size
            action_size (int): action space size
            seed (int): random seed
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = np.random.seed(seed)
        
        # Hyperparameters
        self.gamma = config["discount_factor"]
        self.epsilon = config["clip_epsilon"]
        self.epochs = config["sgd_epochs"]

        # Networks
        self.net = get_network(config["network"], state_size, action_size, seed)
        self.optimizer = optim.Adam(self.net.parameters(), lr=config["lr"])

        self.t_step = 0
        self.episodes = 0
        self.trajectory = []
    
    def step(self, state, action, reward, next_state, is_done):
        """Saves information for the current trajectory and updates network at the end of the episode
    
        Params
        ======
            state (array): state
            action (int): action
            reward (float): reward
            next_state (array): next state
            is_done (bool): whether the episode has ended
        """

        self.trajectory[-1][3] = reward

        # Learn
        loss = None
        if np.any(is_done):
            loss = self.learn()
            self.episodes += 1
            
        self.t_step += 1

        return loss

    def act(self, state, test=False):
        """Returns action given state
    
        Params
        ======
            state (array): state
            test (bool): whether it is used in training or testing
        """

        state_in = torch.from_numpy(state).float().to(device)
        self.net.eval()
        with torch.no_grad():
            param1, param2 = self.net(state_in)
            action, probability = self.net.sample(param1, param2)

            action = action.detach().cpu().numpy()
            probability = probability.detach().cpu().numpy()
        self.net.train()

        self.trajectory.append([state, action, probability, None])

        if test:
            self.net.eval()
            with torch.no_grad():
                _, mean, _ = self.net.get_probability(state_in, action)
                action = mean
            self.net.train()

        return action

    def learn(self):
        """Updates network
    
        Params
        ======
        """

        states = np.array([t[0] for t in self.trajectory])
        actions = np.array([t[1] for t in self.trajectory])
        old_probs = np.array([t[2] for t in self.trajectory])
        rewards = np.array([t[3] for t in self.trajectory])

        losses = []
        for _ in range(self.epochs):
            loss = -self.ppo_loss(old_probs, states, actions, rewards)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()

            losses.append(loss.detach().cpu().numpy())
            del loss

        self.epsilon *= .999
        self.net.update_var()
        self.trajectory = []

        return np.mean(losses)

    def ppo_loss(self, old_probs, states, actions, rewards):
        """Computes loss for sgd update

        Params
        ======
            old_probs (array): old probabilities of the actions
            states (array): states
            actions (array): selected actions
            rewards (array): rewards
        """

        discounts = np.array([[self.gamma ** i] * rewards[0].shape[0] for i in range(len(rewards))])
        rewards = np.array(rewards) * discounts
        rewards = (np.sum(rewards, axis=0) - np.cumsum(rewards, axis=0) + rewards) / discounts

        rewards_mean = np.expand_dims(np.mean(rewards, axis=1), axis=-1)
        rewards_std = np.expand_dims(np.std(rewards, axis=1), axis=-1)
        rewards_std[rewards_std < 1e-6] = 1
        rewards = (rewards - rewards_mean) / rewards_std



        states = torch.tensor(states, dtype=torch.float, device=device).view(-1, states.shape[-1])
        actions = torch.tensor(actions, dtype=torch.float, device=device).view(-1, actions.shape[-1])
        rewards = torch.tensor(rewards, dtype=torch.float, device=device).view(-1)
        old_probs = torch.tensor(old_probs, dtype=torch.float, device=device).view(-1)


        # convert states to policy (or probability)
        new_probs, mean, var = self.net.get_probability(states, actions)

        ratio = new_probs / (old_probs + 1e-6)
        ratio_clip = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)

        return torch.mean(rewards * torch.min(ratio, ratio_clip))
            
    def save(self, name):
        """Saves network parameters
    
        Params
        ======
            name (str): method name
        """
        
        torch.save(self.net.state_dict(), f"./data/{name}/checkpoint.pth")
    
    def load(self, name):
        """Loads network parameters
    
        Params
        ======
            name (str): method name
        """
        
        self.net.load_state_dict(torch.load(f"./data/{name}/checkpoint.pth", map_location=device))
