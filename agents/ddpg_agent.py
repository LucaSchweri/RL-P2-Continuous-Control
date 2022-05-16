import numpy as np
import random
from collections import deque
import json

import torch
import torch.nn.functional as F
import torch.optim as optim

from importlib import reload
import networks.networks
reload(networks.networks)
from networks.networks import get_network

device = "cuda" if torch.cuda.is_available() else "cpu"

class DDPGAgent():
    """DDPG Agent
    
    Configurations (config.json)
    ======
        actor_network (str): name of actor network
        critic_network (str): name of critic network
        actor_lr (float): learning rate of actor network
        critic_lr (float): learning rate of critic network
        buffer_size (int): size of replay buffer
        batch_size (int): batch size
        update_net_steps (int): every "update_net_steps" steps the network updates its parameters
        repeated_update (int): repeat the network update this many times
        discount_factor (float): discount factor
        target_ema (float): exponential moving average parameter for the target network
        double_q_learning (bool): whether or not the agent should use double Q-learning
        prioritized_experience_replay (bool): whether or not the agent should use prioritized experience replay
        n_step_bootstrapping: parameter for n-step bootstrapping
    """

    def __init__(self, config, state_size, action_size, seed):
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
        self.batch_size = config["batch_size"]
        self.update_net_steps = config["update_net_steps"]
        self.tau = config["target_ema"]
        self.gamma = config["discount_factor"]
        self.n_bootstrapping = config["n_step_bootstrapping"]
        self.repeated_update = config["repeated_update"]

        # Networks
        self.actor_net = get_network(config["actor_network"], state_size, action_size, seed)
        self.target_actor_net = get_network(config["actor_network"], state_size, action_size, seed)
        self.critic_net = get_network(config["critic_network"], state_size, action_size, seed)
        self.target_critic_net = get_network(config["critic_network"], state_size, action_size, seed)
        self.optimizer_actor = optim.Adam(self.actor_net.parameters(), lr=config["actor_lr"])
        self.optimizer_critic = optim.Adam(self.critic_net.parameters(), lr=config["critic_lr"])
        self.target_actor_net.eval()
        self.target_critic_net.eval()

        # Replay Memory
        self.replay_buffer = ReplayBuffer(config["buffer_size"], self.batch_size, seed)

        self.t_step = 0
        self.episodes = 0
        self.trajectory_buffer = deque(maxlen=int(self.n_bootstrapping))
        self.losses = []

    def step(self, state, action, reward, next_state, is_done):
        """Saves SARS tuple in replay buffer and updates network
    
        Params
        ======
            state (array): state
            action (int): action
            reward (float): reward
            next_state (array): next state
            is_done (bool): whether the episode has ended
        """

        self.trajectory_buffer.append([state, action, reward, next_state, is_done])

        final_loss = None
        if len(self.trajectory_buffer) >= self.n_bootstrapping:
            # Save Experience
            cumulative_reward = np.sum(np.array([[self.gamma ** i] for i in range(self.n_bootstrapping)]) * np.array([elem[2] for elem in self.trajectory_buffer]), axis=0)
            self.replay_buffer.add(self.trajectory_buffer[0][0], self.trajectory_buffer[0][1], cumulative_reward, next_state, is_done, np.full_like(is_done, self.n_bootstrapping))

            if np.any(is_done):
                while len(self.trajectory_buffer) > 1:
                    self.trajectory_buffer.popleft()
                    cumulative_reward = np.sum(
                        np.array([[self.gamma ** i] for i in range(len(self.trajectory_buffer))]) * np.array([elem[2] for elem in self.trajectory_buffer]), axis=0)
                    self.replay_buffer.add(self.trajectory_buffer[0][0], self.trajectory_buffer[0][1], cumulative_reward, next_state, is_done, np.full_like(is_done, len(self.trajectory_buffer)))

            # Learn
            if (self.t_step + 1) % self.update_net_steps == 0 and len(self.replay_buffer) >= self.batch_size:
                loss = self.learn()
                self.losses.append(loss)
            
        if np.any(is_done):
            self.episodes += 1
            self.trajectory_buffer.clear()
            final_loss = np.mean(self.losses)
            self.losses = []
            
        self.t_step += 1

        return final_loss

    def act(self, state, test=False):
        """Returns action given state
    
        Params
        ======
            state (array): state
            test (bool): whether it is used in training or testing
        """

        state_in = torch.from_numpy(state).float().to(device)
        self.actor_net.eval()
        with torch.no_grad():
            action = self.actor_net(state_in, add_noise=(not test))

            action = action.detach().cpu().numpy()
        self.actor_net.train()

        return action

    def learn(self):
        """Experience replay learning

        Params
        ======
        """

        losses = []
        for _ in range(self.repeated_update):
            states, actions, rewards, next_states, dones, counts = self.replay_buffer.sample()

            self.target_actor_net.eval()
            self.target_critic_net.eval()
            with torch.no_grad():
                target = rewards + self.gamma**counts * (1 - dones) * self.target_critic_net(torch.cat([next_states, self.target_actor_net(next_states)], dim=1))

            prediction = self.critic_net(torch.cat([states, actions], dim=1))

            loss_fcn = torch.nn.MSELoss()
            loss_critic = loss_fcn(prediction, target)

            self.optimizer_critic.zero_grad()
            loss_critic.backward()
            self.optimizer_critic.step()

            action_gain = torch.mean(self.critic_net(torch.cat([states, self.actor_net(states)], dim=1)))
            loss_actor = -action_gain

            self.optimizer_actor.zero_grad()
            loss_actor.backward()
            self.optimizer_actor.step()

            losses.append(loss_critic.detach().cpu().numpy() + loss_actor.detach().cpu().numpy())

            # ------------------- update target network ------------------- #
            self.update_target()

        return np.mean(losses)

    def update_target(self):
        """Exponential moving average of target network
    
        Params
        ======
        """

        for target_param, param in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
            target_param.data.copy_(self.tau*param.data + (1.0-self.tau)*target_param.data)

        for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
    def save(self, name):
        """Saves network parameters
    
        Params
        ======
            name (str): method name
        """
        
        torch.save(self.actor_net.state_dict(), f"./data/{name}/actor_checkpoint.pth")
        torch.save(self.critic_net.state_dict(), f"./data/{name}/critic_checkpoint.pth")
        torch.save(self.target_actor_net.state_dict(), f"./data/{name}/target_actor_checkpoint.pth")
        torch.save(self.target_critic_net.state_dict(), f"./data/{name}/target_critic_checkpoint.pth")
    
    def load(self, name):
        """Loads network parameters
    
        Params
        ======
            name (str): method name
        """
        
        self.actor_net.load_state_dict(torch.load(f"./data/{name}/actor_checkpoint.pth", map_location=device))
        self.critic_net.load_state_dict(torch.load(f"./data/{name}/critic_checkpoint.pth", map_location=device))
        self.target_actor_net.load_state_dict(torch.load(f"./data/{name}/target_actor_checkpoint.pth", map_location=device))
        self.target_critic_net.load_state_dict(torch.load(f"./data/{name}/target_critic_checkpoint.pth", map_location=device))

class ReplayBuffer:
    """Normal replay buffer
    """

    def __init__(self, buffer_size, batch_size, seed):
        """Initalizes replay buffer
    
        Params
        ======
            action_size (int): action size
            buffer_size (int): buffer size
            batch_size (int): batch size
            seed (int): random seed
        """
        
        self.buffer = deque(maxlen=int(buffer_size))  
        self.batch_size = batch_size
        self.seed = np.random.seed(seed)
    
    def add(self, state, action, reward, next_state, done, count):
        """Adds tuple to replay buffer
    
        Params
        ======
            state (array): state
            action (int): action
            reward (int): reward
            next_state (array): next state
            done (bool): whether th episode has ended
        """
        
        elem = [state, action, reward, next_state, done, count]
        self.buffer.append(elem)
    
    def sample(self):
        """Get a experience sample from the replay buffer
    
        Params
        ======
        """
        
        idx = np.random.choice([i for i in range(len(self.buffer))], size=self.batch_size)
        
        experiences = [self.buffer[int(i)] for i in idx]

        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).float().to(device)
        rewards = torch.from_numpy(np.vstack(np.expand_dims([e[2] for e in experiences], axis=-1))).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack(np.expand_dims([e[4] for e in experiences], axis=-1)).astype(np.uint8)).float().to(device)
        counts = torch.from_numpy(np.vstack(np.expand_dims([e[5] for e in experiences], axis=-1)).astype(np.uint8)).float().to(device)
  
        return states, actions, rewards, next_states, dones, counts

    def __len__(self):
        return len(self.buffer)