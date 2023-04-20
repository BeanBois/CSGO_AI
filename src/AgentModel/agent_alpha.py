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
    
    def push(self, seq_obs, seq_p_obs, seq_actions, seq_reward, seq_done, goal_info):
        tmp = seq_obs
        tmp1 = seq_p_obs
        seq_obs = [flatten_obs(obs) for obs in tmp[:-1]]
        seq_p_obs = [flatten_p_obs(p_obs) for p_obs in tmp1[:-1]]
        seq_next_obs = [flatten_obs(obs) for obs in tmp[1:]]
        seq_next_p_obs = [flatten_p_obs(p_obs) for p_obs in tmp1[1:]]
        seq_goal = [flatten_goal(goal['goal state']) for goal in goal_info]
        seq_p_goal = [flatten_goal(goal['partial goal state']) for goal in goal_info]
        transition = tuple((seq_obs, seq_p_obs, seq_actions, seq_reward, seq_next_obs, seq_next_p_obs, seq_goal, seq_p_goal,seq_done))
        self.buffer.append(transition)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def sample(self, batch_size):
        arr = np.asarray(random.sample(self.buffer, batch_size)) #shape is (batch_size, seq_len, feature_size)
        #we want batches
        state_batch, p_state_batch, action_batch, reward_batch, next_state_batch, next_p_state_batch, goal_batch, p_goal_batch, done_batch = [], [], [], [], [], [], [], [], []
        for transitions in arr:
            state_batch.append(transitions[0])
            p_state_batch.append(transitions[1])
            action_batch.append(transitions[2])
            reward_batch.append(transitions[3])
            next_state_batch.append(transitions[4])
            next_p_state_batch.append(transitions[5])
            goal_batch.append(transitions[6])
            p_goal_batch.append(transitions[7])
            done_batch.append(transitions[8])
        
        return state_batch, p_state_batch, action_batch, reward_batch, next_state_batch, next_p_state_batch, goal_batch, p_goal_batch, done_batch
    
    def process_data(self, samples):
        #takes samples of transition and output a batch of sequential data
        #shape is (batch_size, seq_len, feature_size)
        state_batch = np.zeros((len(samples), len(samples[0]), len(samples[0][0][0])))
        p_state_batch = np.zeros((len(samples), len(samples[0]), len(samples[0][0][1])))
        action_batch = np.zeros((len(samples), len(samples[0]), len(samples[0][0][2])))
        reward_batch = np.zeros((len(samples), len(samples[0]), 1))
        next_state_batch = np.zeros((len(samples), len(samples[0]), len(samples[0][0][4])))
        next_p_state_batch = np.zeros((len(samples), len(samples[0]), len(samples[0][0][5])))
        goal_batch = np.zeros((len(samples), len(samples[0]), 1))
        p_goal_batch = np.zeros((len(samples), len(samples[0]), 1))
        done_batch = np.zeros((len(samples), len(samples[0]), 1))
        for i in range(len(samples)):
            transitions = samples[i]
            for j in range(len(transitions)):
                state_batch[i][j] = transitions[j][0]
                p_state_batch[i][j] = transitions[j][1]
                action_batch[i][j] = transitions[j][2]
                reward_batch[i][j] = transitions[j][3]
                next_state_batch[i][j] = transitions[j][4]
                next_p_state_batch[i][j] = transitions[j][5]
                goal_batch[i][j] = transitions[j][6]
                p_goal_batch[i][j] = transitions[j][7]
                done_batch[i][j] = transitions[j][8]
        return state_batch, p_state_batch, \
                action_batch, reward_batch, \
                next_state_batch, next_p_state_batch,\
                goal_batch, p_goal_batch, done_batch
            
            


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
        self.relu = nn.ReLU()
    def forward(self, x):
        # Initialize the hidden states and cell states
        x = torch.stack(x, dim=0)
        h1 = torch.zeros(self.hidden_size)
        c1 = torch.zeros(self.hidden_size)
        h2 = torch.zeros(self.hidden_size)
        c2 = torch.zeros(self.hidden_size)
        h3 = torch.zeros(self.hidden_size)
        c3 = torch.zeros(self.hidden_size)
        h4 = torch.zeros(self.hidden_size)
        c4 = torch.zeros(self.hidden_size)
        
        # Iterate through the input sequence
        for i in range(x.size(0)):
            # Apply the first LSTM cell
            h1, c1 = self.cell1(x[i,:], (h1, c1))
            h1, c1 = self.relu(h1), self.relu(c1)
            # Apply the second LSTM cell after two time steps
            if i % 4 == 1:
                h2, c2 = self.cell2(h1, (h2, c2))
                h2, c2 = self.relu(h2), self.relu(c2)
            # Apply the third LSTM cell after four time steps
            if i % 4 == 2:
                h3, c3 = self.cell3(h2, (h3, c3))
                h3, c3 = self.relu(h3), self.relu(c3)
            # Apply the fourth LSTM cell after eight time steps
            if i % 4 == 3:
                h4, c4 = self.cell4(h3, (h4, c4))
                h4, c4 = self.relu(h4), self.relu(c4)
        # Return the output of the last LSTM cell
        return h4


#used to process Final Actor layer output
class ActionLayer(nn.Module):
    def __init__(self,action_size):
        super(ActionLayer, self).__init__()
        self.action_size = action_size
    def forward(self, x):
        # action = torch.multinomial(x, 1).item()
        # return action
        action = torch.zeros(self.action_size)
        for i in range(self.action_size):
            rand = torch.rand(1)
            if rand < x[i]:
                action[i] = 1
            else:
                action[i] = 0
        return action

#HOW MANY STEPS TO LOOK AHEAD FOR PLANNING
LOOK_AHEAD = 4

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim):
        super(Critic, self).__init__()
        self.model = MTRNN(flatdim(state_dim) + flatdim(action_dim) + flatdim(goal_dim), 512)
        self.output_latyer = nn.Linear(512, 1)
        self.look_ahead = LOOK_AHEAD
        # self.relu = nn.ReLU()

    def forward(self, seq_state, action, seq_goal):
        x = self._process_input(seq_state, action, seq_goal)
        x = self.model(x)
        x = self.output_layer(x)
        return x

    def _process_input(self, seq_state, seq_action,seq_goal):
        xs = []
        for i in range(self.look_ahead):
            p_s_t = seq_state[i]
            a_t = seq_action[i]
            g_o = seq_goal[i]
            if type(p_s_t) != torch.Tensor:
                p_s_t = to_tensor(p_s_t)
            if type(a_t) != torch.Tensor:
                a_t = to_tensor(a_t.data)
            if a_t.dim() == 0:
                a_t = a_t.unsqueeze(0)
            if type(g_o) != torch.Tensor:
                g_o = to_tensor(g_o)
            x = torch.cat((p_s_t, a_t, g_o), dim=0)
            xs.append(x)
        return xs

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim):
        super(Actor, self).__init__()
        self.model = MTRNN(flatdim(state_dim) + flatdim(goal_dim), 512)
        self.output_layer = nn.Linear(512, flatdim(action_dim))
        self.action_layer = ActionLayer(flatdim(action_dim))
        #init weight
        self.look_ahead = LOOK_AHEAD
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()

    def _process_input(self, seq_state, seq_goal):
        xs = []
        for i in range(self.look_ahead):
            p_s_t = seq_state[i]
            g_o = seq_goal[i]
            if type(p_s_t) != torch.Tensor:
                p_s_t = flatten_p_obs(p_s_t)
                p_s_t = to_tensor(p_s_t)
            if type(g_o) != torch.Tensor:
                g_o = flatten_goal(g_o)
                g_o = to_tensor(g_o)
            x = torch.cat((p_s_t, g_o), dim=0)
            xs.append(x)
        return xs
    
    def forward(self, state, goal):
        x = self._process_input(state, goal)
        x = self.model(x)
        x = self.output_layer(x)
        x = self.relu(x)
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
        self.look_ahead = LOOK_AHEAD
        self.discount = 0.98
        self.tau = 0.98
        self.batch_size = 5
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
        
            #get sequence of XXX from batch data
            print('state_batch[i]', state_batch[i])
            print('next_state_batch[i]', next_state_batch[i])
            print('p_state_batch[i]', p_state_batch[i])
            print('next_p_state_batch[i]', next_p_state_batch[i])
            
            next_q_values = self.target_critic(
                to_tensor(next_state_batch[i],is_batch=True),
                self.target_actor(to_tensor(next_p_state_batch[i], is_batch=True), to_tensor(p_goal_batch[i], is_batch=True)), #need to return a sequence of actions, but it is not meant to do this
                to_tensor(goal_batch[i], is_batch=True),
            )
            next_q_values.volatile=False

            target_q_batch = torch.tensor(reward_batch[i][-1]) + \
                self.discount*torch.tensor(int(terminal_batch[i][-1]))*next_q_values

            # Critic update
            self.critic.zero_grad()

            q_batch = self.critic( to_tensor(state_batch[i], is_batch=True), to_tensor(action_batch[i],is_batch=True), to_tensor(goal_batch[i],is_batch=True) )
            # q_batch = q_batch.to(torch.long)
            # target_q_batch = target_q_batch.to(torch.long)
            value_loss = criterion(q_batch, target_q_batch)
            value_loss.backward()
            self.critic_optimizer.step()

            # Actor update
            self.actor.zero_grad()

            policy_loss = -self.critic(
                to_tensor(state_batch[i], is_batch=True),
                self.actor(to_tensor(p_state_batch[i], is_batch=True), to_tensor(p_goal_batch[i], is_batch=True)),
                to_tensor(goal_batch[i], is_batch=True)
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

    def observe(self, seq_obs, seq_p_obs, action, seq_reward, seq_done, goal_info):
        if True:
            self.memory.push(seq_obs, seq_p_obs, action, seq_reward, seq_done, goal_info)
            self.s_t = seq_obs[-1]
            self.p_s_t = seq_p_obs[-1]

    #redo this
    #we just gonna generate a random number between 0 and 2**6
    #and then we gonna convert it to a binary number
    #and then we gonna convert it to a list of 0 and 1
    def random_action(self):
        # actions = []
        # for i in range(self.look_ahead):
        action = np.random.uniform(0, 1, size=self.action_dim)
        action = (action > 0.5)
        action = 1*action
        self.a_t = action
        action = self._process_action(action)
            # actions.append(action)
        return action
        # return actions

    #redo this
    #output of actor network is
    def select_action(self, seq_p_s_t, seq_g_o):
        prob = np.random.uniform(0, 1)
        if prob < self.exploration_prob:
            return self.random_action()
        action = self.actor(seq_p_s_t,seq_g_o)
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

    def load_weights(self, output='CSGO_model_weight_alpha',epoch_num = 0, epoch_cycle_num = 0):
        if output is None: return

        self.actor.load_state_dict(
            torch.load( '{}/actor_{}_{}.pkl'.format(output,epoch_num,epoch_cycle_num))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic_{}_{}.pkl'.format(output,epoch_num,epoch_cycle_num,))
        )

    def save_model(self,output='CSGO_model_weights_alpha',epoch_num = 0, epoch_cycle_num = 0):
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
