import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from .util import *
from gym.spaces.utils import flatdim

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
    
    def push(self, state, p_state, action, reward, next_state, next_p_state, goal, p_goal, done):
        transition = (state, p_state, action, reward, next_state, next_p_state, goal, p_goal, done)
        self.buffer.append(transition)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def sample(self, batch_size):
        state_batch, p_state_batch, action_batch, reward_batch, next_state_batch, next_p_state_batch, goal_batch, p_goal_batch, done_batch = zip(*random.sample(self.buffer, batch_size))
        return torch.tensor(state_batch), torch.to_tensor(p_state_batch), \
                torch.tensor(action_batch), torch.tensor(reward_batch).unsqueeze(1), \
                torch.tensor(next_state_batch), torch.to_tensor(next_p_state_batch),\
                torch.to_tensor(goal_batch), torch.to_tensor(p_goal_batch), torch.tensor(done_batch).unsqueeze(1)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(flatdim(state_dim) + flatdim(action_dim) + flatdim(goal_dim), 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action, goal):
        x = torch.cat([state, action, goal], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(flatdim(state_dim) + flatdim(goal_dim), 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, flatdim(action_dim)) # action_dim = 9
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, state, goal):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x


class DDPG:
    def __init__(self, state_dim, action_dim, goal_dim, device):

        self.device = device
        self.actor = Actor(state_dim, action_dim,goal_dim).to(device)
        self.critic = Critic(state_dim, action_dim, goal_dim).to(device)

        self.target_actor = Actor(state_dim, action_dim,goal_dim).to(device)
        self.target_critic = Critic(state_dim, action_dim, goal_dim).to(device)

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
        self.exploration_noise = 0.05 

        hard_update(self.target_actor, self.actor) # Make sure target is with the same weight
        hard_update(self.target_critic, self.critic)
        
        #Create replay buffer
        self.memory = ReplayBuffer(max_size=10000)

        
        self.s_t = None # Most recent state
        self.p_s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.gs = None # Most recent goal
        self.go = None # Most recent partial goal
        # self.is_training = True

        # 
        # if USE_CUDA: self.cuda()

    def update_policy(self):
        # Sample batch
        state_batch, p_state_batch ,action_batch,\
        reward_batch, next_state_batch, next_p_state_batch,\
        goal_batch, p_goal_batch , terminal_batch = self.memory.sample(self.batch_size)

        # Prepare for the target q batch
        next_q_values = self.target_critic([
            to_tensor(next_state_batch, volatile=True),
            self.target_actor(to_tensor(next_state_batch, volatile=True), to_tensor(p_goal_batch), volatile=True),
            to_tensor(goal_batch, volatile=True),
        ])
        next_q_values.volatile=False

        target_q_batch = to_tensor(reward_batch) + \
            self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([ to_tensor(state_batch), to_tensor(action_batch), to_tensor(goal_batch) ])
        
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch), to_tensor(p_goal_batch)),
            to_tensor(goal_batch)
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)
        return value_loss.data[0], policy_loss.data[0]

    def eval(self):
        self.actor.eval()
        self.target_actor.eval()
        self.critic.eval()
        self.target_critic.eval()

    def cuda(self):
        self.actor.cuda()
        self.target_actor.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def observe(self, s_t, p_s_t, a_t, r_t, s_t1, p_st1, gs, go, done):
        if self.is_training:
            self.memory.push(s_t, p_s_t, a_t, r_t, s_t1, p_st1, gs, go ,done)
            self.s_t = s_t1
            self.p_s_t = p_st1

    #redo this
    #we just gonna generate a random number between 0 and 2**6
    #and then we gonna convert it to a binary number
    #and then we gonna convert it to a list of 0 and 1
    def random_action(self):
        action = np.random.uniform(0, 1, size=action_dim)
        action = (action > 0.5).int()
        self.a_t = action
        return self._process_action(action)

    #redo this
    #output of actor network is
    def select_action(self, p_s_t, g_o, decay_epsilon=True):
        action = self.actor(to_tensor(np.array([p_s_t])), to_tensor(np.array([g_o])))
        action = (action > 0.5).int()
        self.a_t = action
        return self._process_action(action)

    def _process_action(self,action):
        return(action[0], action[1], action[2], action[3],action[4],
            action[5], action[6])

    def reset(self, obs, p_obs, goal, p_goal):
        self.s_t = obs
        self.p_s_t = p_obs
        self.a_t = None
        self.gs = goal
        self.go = p_goal

    def load_weights(self, output='csgo_model_weights'):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )

    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def seed(self,s):
        torch.manual_seed(s)
