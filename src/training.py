import os
import time
from collections import deque
import pickle

import pandas as pd
# from baselinesaddpg.ddpg import DDPG
from AgentModel.agent import DDPG
# from AgentModel.util import mpi_mean, mpi_std, mpi_max, mpi_sum
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# from AgentModel import logger
import numpy as np
# import tensorflow as tf
# from mpi4py import MPI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# code based on https://github.com/openai/baselines
TRAINING_STATS_SAVE_FILEPATH = '/Users/beep_kai/Desktop/fyp/CSGO_AI_ref/src/training_statistics/epoch_data/training_data/'
EVALUATION_STATS_SAVE_FILEPATH = '/Users/beep_kai/Desktop/fyp/CSGO_AI_ref/src/training_statistics/epoch_data/evaluation_data/'
TESTING_STATS_SAVE_FILEPATH = '/Users/beep_kai/Desktop/fyp/CSGO_AI_ref/src/testing_data/'
def train(env, nb_epochs = 40, nb_epoch_cycles = 20, nb_train_steps = 100, nb_of_rounds = 5, eval_env=None, load_model = True):
    print('start training')
    print('init agent and env')
    
    restarting_epoch = False
    restarting_epoch_cycle = False
    agent = DDPG(env.observation_space, env.action_space, env.goal_space, device)
    # obs, p_obs, reward, done, goal, p_goal = env.reset()
    # agent.reset(obs, p_obs, goal, p_goal)
    if load_model:
        LAST_EPOCH = 30#update this value
        LAST_EPOCH_CYCLE = 5#update this value
        agent.load_weights(epoch_num=LAST_EPOCH, epoch_cycle_num=LAST_EPOCH_CYCLE)
        restarting_epoch = True
        restarting_epoch_cycle = True
        print('model loaded')
    start_epoch = 0
    
    if restarting_epoch:
        start_epoch = LAST_EPOCH 
        restarting_epoch = False
    
    for epoch in range(start_epoch,nb_epochs):
        #init data structure here to collectn data foreach epoch
        avg_reward_in_epoch = []
        avg_winrate_in_epoch= []
        start = 0
        if restarting_epoch_cycle:
            start += LAST_EPOCH
            restarting_epoch_cycle = False
        for epoch_cycle in range(start,nb_epoch_cycles):
            #init data structure here to collectn data foreach epoch cycle
            cum_reward_in_epoch_cycle = 0
            rounds_won = 0
            for round in range(nb_of_rounds):
                obs, p_obs, reward, done, goal, p_goal = env.reset()
                agent.reset(obs, p_obs, goal, p_goal)
                #init data structure here to collect data for each round
                round_winner = None
                episode_reward = 0
                episode_step = 0
                t=0
                while True:
                    print("round number : ", round)
                    print("epoch number : ", epoch)
                    print("epoch cycle number : ", epoch_cycle)
                    print('rounds won : ', rounds_won)
                    action, q = None, 0
                    if epoch < 30:
                        action, q = agent.random_action(), 0
                    else:
                        action, q = agent.select_action(p_obs, p_goal), 0
                    obs = env.get_current_observation()
                    new_obs, new_p_obs, r, done, info = env.step(action)
                    goal, p_goal = info['goal state'], info['partial goal state']
                    t += 1
                    episode_reward += r
                    episode_step += 1
                    agent.observe(obs, p_obs, action, r, new_obs, new_p_obs, goal, p_goal ,done)
                    p_obs = new_p_obs
                    
                    if done:
                        
                        #collecting data
                        cum_reward_in_epoch_cycle += episode_reward
                        print(f"winner : {obs['winner']} " )
                        if new_obs['winner'] == 1:
                            rounds_won += 1
                        

                        break
            
            #collecting data
            avg_reward_in_epoch.append(cum_reward_in_epoch_cycle/nb_of_rounds) #avg reward in epoch cycle
            avg_winrate_in_epoch.append(float(rounds_won)/nb_of_rounds) #win rate in epoch cycle, out of 5 rounds
            
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


def test(env,nb_of_rounds=100):
    agent = DDPG(env.observation_space, env.action_space, env.goal_space, device)
    print('start training')
    print('init agent and env')
    
    restarting_epoch = False
    restarting_epoch_cycle = False
    agent = DDPG(env.observation_space, env.action_space, env.goal_space, device)
    # obs, p_obs, reward, done, goal, p_goal = env.reset()
    # agent.reset(obs, p_obs, goal, p_goal)
    LAST_EPOCH = 31#update this value
    LAST_EPOCH_CYCLE = 3 #update this value
    agent.load_weights(epoch_num=LAST_EPOCH, epoch_cycle_num=LAST_EPOCH_CYCLE)
    
    rewards=[]
    rounds_won = 0
    for round in range(nb_of_rounds):
        obs, p_obs, reward, done, goal, p_goal = env.reset()
        agent.reset(obs, p_obs, goal, p_goal)
        #init data structure here to collect data for each round
        episode_reward = 0
        episode_step = 0
        rounds_won = []
        t=0
        while True:
            print("round number : ", round)
            action, q = None, 0
            action, q = agent.select_action(p_obs, p_goal), 0
            obs = env.get_current_observation()
            new_obs, new_p_obs, r, done, info = env.step(action)
            goal, p_goal = info['goal state'], info['partial goal state']
            t += 1
            episode_reward += r
            episode_step += 1
            agent.observe(obs, p_obs, action, r, new_obs, new_p_obs, goal, p_goal ,done)
            p_obs = new_p_obs
            
            if done:
                
                #collecting data
                rewards.append(episode_reward)
                rounds_won.append(obs['winner'])
                break
        np.save(f'{TESTING_STATS_SAVE_FILEPATH}rewards.npy', np.array(rewards))
        np.save(f'{TESTING_STATS_SAVE_FILEPATH}rounds_won.npy', np.array(rounds_won))
        
