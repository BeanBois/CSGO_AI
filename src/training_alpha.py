import os
import time
from collections import deque
import pickle

import pandas as pd
# from baselinesaddpg.ddpg import DDPG
from AgentModel.agent_alpha import DDPG
# from AgentModel.util import mpi_mean, mpi_std, mpi_max, mpi_sum
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# from AgentModel import logger
import numpy as np
# import tensorflow as tf
# from mpi4py import MPI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAINING_STATS_SAVE_FILEPATH = '/Users/beep_kai/Desktop/fyp/CSGO_AI_ref/src/training_statistics_alpha/epoch_data/training_data/'
EVALUATION_STATS_SAVE_FILEPATH = '/Users/beep_kai/Desktop/fyp/CSGO_AI_ref/src/training_statistics_alpha/epoch_data/evaluation_data/'

# code based on https://github.com/openai/baselines
def train(env, nb_epochs = 40, nb_epoch_cycles = 20, nb_train_steps = 100, nb_of_rounds = 1, eval_env=None, load_model = False):
    print('start training')
    print('init agent and env')
    
    restarting_epoch = False
    restarting_epoch_cycle = False
    agent = DDPG(env.observation_space, env.action_space, env.goal_space, device)
    # init_actions = [agent.random_action() for _ in range(agent.look_ahead)]
    # seq_obs, seq_p_obs, seq_reward, seq_done, seq_goal_state = env.reset(init_actions)
    # agent.reset(seq_obs[-1], seq_p_obs[-1], seq_reward[-1], seq_done[-1], seq_goal_state[-1])
    if load_model:
        LAST_EPOCH = 1 #update this value
        LAST_EPOCH_CYCLE = 35 #update this value
        agent.load_weights(epoch_num=LAST_EPOCH, epoch_cycle_num=LAST_EPOCH_CYCLE)
        # restarting_epoch = True
        # restarting_epoch_cycle = True
    
    
    
    for epoch in range(nb_epochs):
        #init data structure here to collectn data foreach epoch
        avg_reward_in_epoch = []
        avg_winrate_in_epoch= []
        if restarting_epoch:
            epoch += LAST_EPOCH
            restarting_epoch = False
        for epoch_cycle in range(nb_epoch_cycles):
            #init data structure here to collectn data foreach epoch cycle
            cum_reward_in_epoch_cycle = 0
            rounds_won = 0
            if restarting_epoch_cycle:
                epoch_cycle += LAST_EPOCH_CYCLE + 1
            for round in range(nb_of_rounds):
                seq_actions = [env.idle_action() for i in range(env.look_ahead)]
                seq_obs, seq_p_obs, seq_reward, seq_done, seq_goal_state = env.reset()
                agent.reset(seq_obs[-1], seq_p_obs[-1], seq_goal_state[-1]['goal state'], seq_goal_state[-1]['partial goal state'])
                # prev_obs = seq_obs[-1]
                # prev_p_obs = seq_p_obs[-1]
                # prev_reward = seq_reward[-1]
                # prev_seq_goal_state = seq_goal_state[-1]
                # prev_done = seq_done[-1]
                #init data structure here to collect data for each round
                round_winner = None
                episode_reward = 0
                episode_step = 0
                t=0
                while True:
                    print("round number : ", round)
                    print("epoch number : ", epoch)
                    print("epoch cycle number : ", epoch_cycle)
                    action = None
                    if epoch < 4:
                        action = agent.random_action()
                    else:
                        seq_p_goals = [seq_goal_state[i]['partial goal state'] for i in range(len(seq_goal_state))]
                        action= agent.select_action(seq_p_obs, seq_p_goals)
                    obs = env.get_current_observation()
                    new_obs, new_p_obs, r, done, info = env.step(action)
                    

                    #add new data
                    seq_obs.append(new_obs)
                    seq_p_obs.append(new_p_obs)
                    
                    #removing old data and new data
                    seq_reward.pop(0)
                    seq_reward.append(r)
                    seq_done.pop(0)
                    seq_done.append(done)
                    seq_goal_state.pop(0)
                    seq_goal_state.append(info)     
                    
                    seq_actions.pop(0)
                    seq_actions.append(action)
                    
                    agent.observe(seq_obs, seq_p_obs, seq_actions, seq_reward, seq_done, seq_goal_state)
                    
                    #remove oldest data
                    seq_obs.pop(0)
                    seq_p_obs.pop(0)
                    
                    episode_reward += r
                    t+=1
                    episode_step += 1
                    
                    if done:
                        #collecting data
                        cum_reward_in_epoch_cycle += episode_reward
                        if obs['winner'] == 1:
                            rounds_won += 1
                        

                        break
            
            #collecting data
            avg_reward_in_epoch.append(cum_reward_in_epoch_cycle/nb_of_rounds)
            avg_winrate_in_epoch.append(float(rounds_won)/nb_of_rounds)
            
            #pause game here if you wanna
            env.pause_game()
            #init data structure here to collectn data foreach evaluation in epoch cycle
            epoch_actor_losses = []
            epoch_critic_losses = []
            for _ in range(nb_train_steps):
                cl, al = agent.update_policy()
                epoch_actor_losses.append(al)
                epoch_critic_losses.append(cl)
            
            #save evaluation data
            epoch_actor_losses = np.array(epoch_actor_losses)
            epoch_critic_losses = np.array(epoch_critic_losses)
            np.save(f'{EVALUATION_STATS_SAVE_FILEPATH}epoch_{epoch}_epoch_cycle_{epoch_cycle}_actor_losses.npy', epoch_actor_losses)
            np.save(f'{EVALUATION_STATS_SAVE_FILEPATH}epoch_{epoch}_epoch_cycle_{epoch_cycle}_critic_losses.npy', epoch_critic_losses)
            agent.save_model(epoch_num=epoch, epoch_cycle_num=epoch_cycle)
        np.save(f'{TRAINING_STATS_SAVE_FILEPATH}epoch_{epoch}_avg_reward.npy', np.array(avg_reward_in_epoch))
        np.save(f'{TRAINING_STATS_SAVE_FILEPATH}epoch_{epoch}_avg_winrate.npy', np.array(avg_winrate_in_epoch))    
        restarting_epoch_cycle = False
    
