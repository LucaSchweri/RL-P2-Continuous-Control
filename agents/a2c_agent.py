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

class A2CAgent():
    """A2C Agent
    
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
        clip_epsilon (float): clip of the probability ratio in the surrogate function
        target_ema (float): exponential moving average parameter for the target network
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
        self.epsilon = config["clip_epsilon"]
        self.repeated_update = config["repeated_update"]

        # Networks
        self.actor_net = get_network(config["actor_network"], state_size, action_size, seed)
        self.critic_net = get_network(config["critic_network"], state_size, action_size, seed)
        self.target_critic_net = get_network(config["critic_network"], state_size, action_size, seed)
        self.optimizer_actor = optim.Adam(self.actor_net.parameters(), lr=config["actor_lr"])
        self.optimizer_critic = optim.Adam(self.critic_net.parameters(), lr=config["critic_lr"])
        self.target_critic_net.eval()

        # Replay Memory
        self.replay_buffer = ReplayBuffer(config["buffer_size"], self.batch_size, seed)

        self.t_step = 0
        self.episodes = 0
        self.trajectory_buffer = deque(maxlen=int(self.n_bootstrapping + self.batch_size - 1))
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

        self.trajectory_buffer[-1][3] = reward
        self.trajectory_buffer[-1][4] = next_state
        self.trajectory_buffer[-1][5] = is_done

        final_loss = None
        if len(self.trajectory_buffer) >= self.n_bootstrapping:
            # Save Experience
            cumulative_reward = np.sum(np.array([[self.gamma ** i] for i in range(self.n_bootstrapping)]) * np.array([elem[3] for elem in list(self.trajectory_buffer)[-self.n_bootstrapping:]]), axis=0)
            self.replay_buffer.add(self.trajectory_buffer[-self.n_bootstrapping][0], cumulative_reward, next_state, is_done, np.full_like(is_done, self.n_bootstrapping))

            # Learn
            if (self.t_step + 1) % self.update_net_steps == 0 and len(self.replay_buffer) >= self.batch_size:
                self.learn_critic()
            if (self.t_step + 1) % self.update_net_steps == 0 and len(self.trajectory_buffer) >= self.batch_size + self.n_bootstrapping - 1:
                loss = self.learn_actor()
                self.losses.append(loss)
            
        if np.any(is_done):
            self.episodes += 1
            self.trajectory_buffer.clear()
            self.epsilon *= .999
            self.actor_net.update_var()
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
            param1, param2 = self.actor_net(state_in)
            action, probability = self.actor_net.sample(param1, param2)

            action = action.detach().cpu().numpy()
            probability = probability.detach().cpu().numpy()
        self.actor_net.train()

        self.trajectory_buffer.append([state, action, probability, None, None, None])

        if test:
            self.actor_net.eval()
            with torch.no_grad():
                _, mean, _ = self.actor_net.get_probability(state_in, action)
                action = mean
            self.actor_net.train()

        return action

    def learn_critic(self):
        """Updates critic network

        Params
        ======
        """

        for _ in range(self.repeated_update):
            states, rewards, next_states, dones, counts = self.replay_buffer.sample()

            with torch.no_grad():
                target = rewards + (self.gamma**counts) * self.target_critic_net(next_states)
                target[dones.bool()] = rewards[dones.bool()]

            prediction = self.critic_net(states)

            loss_fcn = torch.nn.MSELoss()
            loss = loss_fcn(prediction, target)

            self.optimizer_critic.zero_grad()
            loss.backward()
            self.optimizer_critic.step()

            # ------------------- update target network ------------------- #
            self.update_target()

    def learn_actor(self):
        """Updates actor network

        Params
        ======
        """
        with torch.no_grad():
            if self.n_bootstrapping == 1:
                buffer_start = list(self.trajectory_buffer)
                buffer_end = list(self.trajectory_buffer)
            else:
                buffer_start = list(self.trajectory_buffer)[:-(self.n_bootstrapping-1)]
                buffer_end = list(self.trajectory_buffer)[(self.n_bootstrapping-1):]
            states = torch.from_numpy(np.array([elem[0] for elem in buffer_start])).float().to(device).view(-1, self.trajectory_buffer[0][0].shape[-1])
            actions = torch.from_numpy(np.array([elem[1] for elem in buffer_start])).float().to(device).view(-1, self.trajectory_buffer[0][1].shape[-1])
            old_probs = torch.from_numpy(np.array([elem[2] for elem in buffer_start])).float().to(device).view(-1)
            discounts = np.array([[[self.gamma ** i] for i in range(self.n_bootstrapping)]])
            rewards = np.array([[elem[3] for elem in list(self.trajectory_buffer)[i:i+self.n_bootstrapping]] for i in range(self.batch_size)])
            rewards = torch.from_numpy(np.sum(discounts * rewards, axis=1)).float().to(device).view(-1)
            next_states = torch.from_numpy(np.array([elem[4] for elem in buffer_end])).float().to(device).view(-1, self.trajectory_buffer[0][4].shape[-1])
            is_done = np.any(self.trajectory_buffer[-1][5])

            if is_done:
                rewards[:-1] = rewards[:-1] + (self.gamma**self.n_bootstrapping) * self.critic_net(next_states[:-1])[:, 0]
            else:
                rewards = rewards + (self.gamma ** self.n_bootstrapping) * self.critic_net(next_states)[:, 0]

            advantages = rewards - self.critic_net(states)[:, 0]

        losses = []
        for _ in range(self.repeated_update):
            new_probs, mean, var = self.actor_net.get_probability(states, actions)

            ratio = new_probs / (old_probs + 1e-6)
            ratio_clip = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)

            action_gain = torch.mean(advantages * torch.min(ratio, ratio_clip))
            loss = - action_gain #+ regularization

            self.optimizer_actor.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), 1.0)
            self.optimizer_actor.step()

            losses.append(loss.detach().cpu().numpy())
            del loss

        return np.mean(losses)

    def update_target(self):
        """Exponential moving average of target network
    
        Params
        ======
        """

        for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(self.tau*param.data + (1.0-self.tau)*target_param.data)
            
    def save(self, name):
        """Saves network parameters
    
        Params
        ======
            name (str): method name
        """
        
        torch.save(self.actor_net.state_dict(), f"./data/{name}/actor_checkpoint.pth")
        torch.save(self.critic_net.state_dict(), f"./data/{name}/critic_checkpoint.pth")
        torch.save(self.target_critic_net.state_dict(), f"./data/{name}/target_checkpoint.pth")
    
    def load(self, name):
        """Loads network parameters
    
        Params
        ======
            name (str): method name
        """
        
        self.actor_net.load_state_dict(torch.load(f"./data/{name}/actor_checkpoint.pth", map_location=device))
        self.critic_net.load_state_dict(torch.load(f"./data/{name}/critic_checkpoint.pth", map_location=device))
        self.target_critic_net.load_state_dict(torch.load(f"./data/{name}/target_checkpoint.pth", map_location=device))

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
    
    def add(self, state, reward, next_state, done, count):
        """Adds tuple to replay buffer
    
        Params
        ======
            state (array): state
            action (int): action
            reward (int): reward
            next_state (array): next state
            done (bool): whether th episode has ended
        """
        
        elem = [state, reward, next_state, done, count]
        self.buffer.append(elem)
    
    def sample(self):
        """Get a experience sample from the replay buffer
    
        Params
        ======
        """
        
        idx = np.random.choice([i for i in range(len(self.buffer))], size=self.batch_size)
        
        experiences = [self.buffer[int(i)] for i in idx]

        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(device)
        rewards = torch.from_numpy(np.vstack(np.expand_dims([e[1] for e in experiences], axis=-1))).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack(np.expand_dims([e[3] for e in experiences], axis=-1)).astype(np.uint8)).float().to(device)
        counts = torch.from_numpy(np.vstack(np.expand_dims([e[4] for e in experiences], axis=-1)).astype(np.uint8)).float().to(device)
  
        return states, rewards, next_states, dones, counts

    def __len__(self):
        return len(self.buffer)