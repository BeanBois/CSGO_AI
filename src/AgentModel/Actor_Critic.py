import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
    
    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def sample(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*random.sample(self.buffer, batch_size))
        return torch.tensor(state_batch), torch.tensor(action_batch), torch.tensor(reward_batch).unsqueeze(1), torch.tensor(next_state_batch), torch.tensor(done_batch).unsqueeze(1)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(state_dim[0], 64, kernel_size=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=2)
        self.fc1 = nn.Linear(256 + np.prod(state_dim[1:]), 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.action_range = action_range

    def forward(self, state, goal):
        x = self.relu(self.conv1(state))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = torch.cat([x.view(x.size(0), -1), goal.view(goal.size(0), -1)], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x)) * self.action_range
        return x


class DDPG:
    def __init__(self, state_dim, action_dim, action_range, device):
        self.device = device
        self.actor = Actor(state_dim, action_dim, action_range).to(device)
        self.critic = Critic(state_dim + action_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim, action_range).to(device)
        self.target_critic = Critic(state_dim + action_dim, action_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        self.replay_buffer = deque(maxlen=105)
        self.discount_factor = 0.98
        self.tau = 0.98
        self.batch_size = 128
        self.horizon = 50
        self.exploration_prob = 0.2
        self.exploration_noise = 0.05 * action_range

    def select_action(self, state, goal):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).