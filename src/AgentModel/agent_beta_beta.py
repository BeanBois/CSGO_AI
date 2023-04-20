import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from .util import *
from gym.spaces.utils import flatdim



#WE NEED SELF.MAX/MIN X,Y,Z COORDINATES FOR THE MAP
#BASICALLY FOR LOCATION WE HAVE A ARRAY FOR EACH AXIS FILLED WITH 1 IF LOCATION IS THERE IN THAT AXIS, ELSE 0
#THEN WE DO A CARTESIAN PRODUCT OF THE 3 ARRAYS TO GET THE LOCATION

CONFIDENCE = 0.7

# criterion = nn.CrossEntropyLoss()
# criterion = nn.NLLLoss()
criterion = nn.MSELoss()
def flatten_location(location,DIMENSIONS):
    x_min, x_max, y_min, y_max, z_min, z_max = DIMENSIONS
    x_axis = np.zeros(x_max-x_min)
    y_axis = np.zeros(y_max-y_min)
    z_axis = np.zeros(z_max-z_min)
    if location is not None:
        x_axis[location[0]-x_min] = 1
        y_axis[location[1]-y_min] = 1
        z_axis[location[2]-z_min] = 1
    return cartesian_product(x_axis,y_axis,z_axis)

    

def flatten_p_obs(obs):
    enemy_pos = obs['enemy']['position'] 
    enemy_loc = enemy_pos['location'] if enemy_pos['location'] is not None else np.array([0,0,0])
    enemy_forw = enemy_pos['forward'] if enemy_pos['forward'] is not None else  np.array([0,0,0])
    enemy_time_seen = enemy_pos['time_seen'] if enemy_pos['time_seen'] is not None else 0
    enemy_health = obs['enemy']['health'] if obs['enemy']['health'] is not None else 100
    enemy_coor_on_screen = obs['enemy']['enemy_screen_coords'] if obs['enemy']['enemy_screen_coords'] is not None else  np.array([0,0])
    
    agent_pos = obs['agent']['position']
    agent_loc = agent_pos['location']
    agent_forw = agent_pos['forward']
    agent_gun = obs['agent']['agent_gun']
    agent_bullets = obs['agent']['agent_bullets']
    agent_health = obs['agent']['health']
    
    bomb_location = obs['bomb_location']['location'] 
    bomb_defusing, time_of_info = obs['bomb_defusing'] 
    if bomb_defusing is None:
        bomb_defusing = 0
    
    
    curr_time = obs['current_time'] if obs['current_time'] > 0 else 0
    winner = obs['winner']
    arr = np.concatenate((enemy_loc, enemy_forw, enemy_time_seen, enemy_health, enemy_coor_on_screen, agent_loc, agent_forw, agent_gun, agent_bullets, agent_health, bomb_location, bomb_defusing, time_of_info, curr_time, winner), axis=None)
    arr=arr.flatten()
    # print(arr)
    return arr

def flatten_obs(p_obs):
    enemy_pos = p_obs['enemy']['position']
    enemy_loc = enemy_pos['location']
    enemy_forw = enemy_pos['forward']
    enemy_time_seen = enemy_pos['time_seen']
    enemy_health = p_obs['enemy']['health']
    enemy_coor_on_screen = p_obs['enemy']['enemy_screen_coords'] if p_obs['enemy']['enemy_screen_coords'] is not None else  np.array([0,0])
    
    agent_pos = p_obs['agent']['position']
    agent_loc = agent_pos['location']
    agent_forw = agent_pos['forward']
    agent_gun = p_obs['agent']['agent_gun']
    agent_bullets = p_obs['agent']['agent_bullets']
    agent_health = p_obs['agent']['health']
    
    bomb_location = p_obs['bomb_location']['location']
    bomb_defusing, time_of_info = p_obs['bomb_defusing']
    
    curr_time = p_obs['current_time'] if p_obs['current_time'] > 0 else 0
    winner = p_obs['winner']
    arr = np.concatenate((enemy_loc, enemy_forw, enemy_time_seen, enemy_health, enemy_coor_on_screen, agent_loc, agent_forw, agent_gun, agent_bullets, agent_health, bomb_location, bomb_defusing, time_of_info, curr_time, winner), axis=None)
    arr = arr.flatten()
    # print(arr)
    return arr

def flatten_goal(goal):
    arr = np.array(goal)
    arr = arr.flatten()
    return arr


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
    
    def push(self, state, p_state, action, reward, next_state, next_p_state, goal, p_goal, done):
        transition = tuple((flatten_obs(state), flatten_p_obs(p_state), action, reward, flatten_obs(next_state), flatten_p_obs(next_p_state), flatten_goal(goal), flatten_goal(p_goal), done))
        self.buffer.append(transition)
        print(len(self.buffer))
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def sample(self, batch_size):
        # state_batch, p_state_batch, action_batch, reward_batch, next_state_batch, next_p_state_batch, goal_batch, p_goal_batch, done_batch = zip(*random.sample(self.buffer, batch_size))
        state_batch, p_state_batch, action_batch, reward_batch, next_state_batch, next_p_state_batch, goal_batch, p_goal_batch, done_batch = np.transpose(random.sample(self.buffer, batch_size))
        # state_batch = np.array([i for i in state_batch])
        # p_state_batch = np.array([i.astype('float') for i in p_state_batch])
        # action_batch = np.array([i for i in action_batch])
        # reward_batch = np.array([i for i in reward_batch])
        # next_state_batch = np.array([i for i in next_state_batch])
        # next_p_state_batch = np.array([i for i in next_p_state_batch])
        # goal_batch = np.array([i for i in goal_batch])
        # p_goal_batch = np.array([i for i in p_goal_batch])
        # done_batch = np.array([i for i in done_batch])
        return state_batch, p_state_batch, \
                action_batch, reward_batch, \
                next_state_batch, next_p_state_batch,\
                goal_batch, p_goal_batch, done_batch
        # return torch.tensor(state_batch), torch.to_tensor(p_state_batch), \
        #         torch.tensor(action_batch), torch.tensor(reward_batch).unsqueeze(1), \
        #         torch.tensor(next_state_batch), torch.to_tensor(next_p_state_batch),\
        #         torch.to_tensor(goal_batch), torch.to_tensor(p_goal_batch), torch.tensor(done_batch).unsqueeze(1)

class MTRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MTRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Define the four LSTM cells with different time constants
        self.cell1 = nn.LSTMCell(input_size, hidden_size)
        self.cell2 = nn.LSTMCell(hidden_size, hidden_size)
        self.cell3 = nn.LSTMCell(hidden_size, hidden_size)
        self.cell4 = nn.LSTMCell(hidden_size, hidden_size)
        
    def forward(self, x):
        # Initialize the hidden states and cell states
        h1 = torch.zeros(x.size(0), self.hidden_size)
        c1 = torch.zeros(x.size(0), self.hidden_size)
        h2 = torch.zeros(x.size(0), self.hidden_size)
        c2 = torch.zeros(x.size(0), self.hidden_size)
        h3 = torch.zeros(x.size(0), self.hidden_size)
        c3 = torch.zeros(x.size(0), self.hidden_size)
        h4 = torch.zeros(x.size(0), self.hidden_size)
        c4 = torch.zeros(x.size(0), self.hidden_size)
        
        # Iterate through the input sequence
        for i in range(x.size(1)):
            # Apply the first LSTM cell
            h1, c1 = self.cell1(x[:,i,:], (h1, c1))
            
            # Apply the second LSTM cell after two time steps
            if i % 2 == 0:
                h2, c2 = self.cell2(h1, (h2, c2))
            
            # Apply the third LSTM cell after four time steps
            if i % 4 == 0:
                h3, c3 = self.cell3(h2, (h3, c3))
            
            # Apply the fourth LSTM cell after eight time steps
            if i % 8 == 0:
                h4, c4 = self.cell4(h3, (h4, c4))
        
        # Return the output of the last LSTM cell
        return h4

class ActionLayer(nn.Module):
    def __init__(self,action_size):
        self.action_size = action_size
    def forward(self, x):
        action = torch.zeros(self.action_size)
        for i in range(self.action_size):
            rand = torch.rand(1)
            if rand < x[i]:
                action[i] = 1
            else:
                action[i] = 0
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(flatdim(state_dim) + flatdim(action_dim) + flatdim(goal_dim), 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 1)
        
        #init weight
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc5.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc6.weight, nonlinearity='relu')
        
        
        # self.relu = nn.ReLU()
        self.relu = torch.nn.functional.relu
        self.sigmoid = nn.Sigmoid()

    def forward(self, state, action, goal):
        x = torch.cat((state.flatten(), action.flatten(), goal.flatten()), dim=0)
        # x = np.concatenate((state.flatten(), action.flatten(), goal.flatten()), axis=None)
        # x = torch.from_numpy(x).float()
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        x = self.relu(x)
        return x


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(flatdim(state_dim) + flatdim(goal_dim), 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, flatdim(action_dim)) # action_dim = 10
        # self.relu = nn.ReLU()
        self.relu = torch.nn.functional.relu
        
        #init weight
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc5.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc6.weight)
        
        
        # self.tanh = nn.Tanh()
        # self.sigmoid = nn.Sigmoid()
        self.action_layer = ActionLayer(flatdim(action_dim))
        

    def forward(self, state, goal):
        x = torch.cat((state.flatten(), goal.flatten()), dim=0)
        # x= np.concatenate((state.flatten(), goal.flatten()), axis=None)
        # x = torch.from_numpy(x).float()
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        x = self.action_layer(x)
        return x


# code based on https://github.com/openai/baselines
class DDPG:
    def __init__(self, state_dim, action_dim, goal_dim, device):
        self.action_dim = flatdim(action_dim)
        self.device = device
        self.actor = Actor(state_dim, action_dim,goal_dim).to(device)
        self.critic = Critic(state_dim, action_dim, goal_dim).to(device)

        self.target_actor = Actor(state_dim, action_dim,goal_dim).to(device)
        self.target_critic = Critic(state_dim, action_dim, goal_dim).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

        self.discount = 0.98
        self.tau = 0.98
        self.batch_size = 100
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
        pl = 0
        vl = 0
        for i in range(self.batch_size):
        # Prepare for the target q batch
            print('state_batch[i]', state_batch[i])
            print('next_state_batch[i]', next_state_batch[i])
            print('p_state_batch[i]', p_state_batch[i])
            print('next_p_state_batch[i]', next_p_state_batch[i])
            
            next_q_values = self.target_critic(
                to_tensor(next_state_batch[i]),
                self.target_actor(to_tensor(next_p_state_batch[i]), to_tensor(p_goal_batch[i])),
                to_tensor(goal_batch[i]),
            )
            next_q_values.volatile=False

            target_q_batch = torch.tensor(reward_batch[i]) + \
                self.discount*torch.tensor(int(terminal_batch[i]))*next_q_values

            # Critic update
            self.critic.zero_grad()

            q_batch = self.critic( to_tensor(state_batch[i]), torch.tensor(action_batch[i]), to_tensor(goal_batch[i]) )
            # q_batch = q_batch.to(torch.long)
            # target_q_batch = target_q_batch.to(torch.long)
            value_loss = criterion(q_batch, target_q_batch)
            value_loss.backward()
            self.critic_optimizer.step()

            # Actor update
            self.actor.zero_grad()

            policy_loss = -self.critic(
                to_tensor(state_batch[i]),
                self.actor(to_tensor(p_state_batch[i]), to_tensor(p_goal_batch[i])),
                to_tensor(goal_batch[i])
            )

            policy_loss = policy_loss.mean()
            policy_loss.backward()
            self.actor_optimizer.step()

            # Target update
            soft_update(self.target_actor, self.actor, self.tau)
            soft_update(self.target_critic, self.critic, self.tau)
            vl += value_loss.data
            policy_loss += pl
            print("value loss: ", vl, "policy loss: ", pl)
        return vl/self.batch_size, pl/self.batch_size

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
        if True:
            self.memory.push(s_t, p_s_t, a_t, r_t, s_t1, p_st1, gs, go ,done)
            self.s_t = s_t1
            self.p_s_t = p_st1

    #redo this
    #we just gonna generate a random number between 0 and 2**6
    #and then we gonna convert it to a binary number
    #and then we gonna convert it to a list of 0 and 1
    def random_action(self):
        action = np.random.uniform(0, 1, size=self.action_dim)
        action = (action > 0.5)
        action = 1*action
        self.a_t = action
        return self._process_action(action)

    #redo this
    #output of actor network is
    def select_action(self, p_s_t, g_o):
        prob = np.random.uniform(0, 1)
        if prob < self.exploration_prob:
            return self.random_action()
        x1 = flatten_p_obs(p_s_t)
        x2 = flatten_goal(g_o)
        x1 = to_tensor(x1)
        x2 = to_tensor(x2)
        action = self.actor(x1,x2)
        print('pre selected action chosen:', action)
        action = (action > CONFIDENCE).int()
        self.a_t = action
        print('action chosen:', action)
        return self._process_action(action)

    def _process_action(self,action):
        return(action[0], action[1], action[2],\
            action[3],action[4], action[5],\
            action[6], action[7], action[8],
            action[9],action[10])

    def reset(self, obs, p_obs, goal, p_goal):
        self.s_t = obs
        self.p_s_t = p_obs
        self.a_t = None
        self.gs = goal
        self.go = p_goal

    def load_weights(self, output='CSGO_model_weights',epoch_num = 0, epoch_cycle_num = 0):
        if output is None: return

        self.actor.load_state_dict(
            torch.load( '{}/actor_{}_{}.pkl'.format(output,epoch_num,epoch_cycle_num))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic_{}_{}.pkl'.format(output,epoch_num,epoch_cycle_num,))
        )

    def save_model(self,output='CSGO_model_weights2',epoch_num = 0, epoch_cycle_num = 0):
        torch.save(
            self.actor.state_dict(),
            '{}/actor_{}_{}.pkl'.format(output,epoch_num,epoch_cycle_num)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic_{}_{}.pkl'.format(output,epoch_num,epoch_cycle_num,)
        )

    def seed(self,s):
        torch.manual_seed(s)
