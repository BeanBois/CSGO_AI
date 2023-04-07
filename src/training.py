import os
import time
from collections import deque
import pickle

import pandas as pd
# from baselines.ddpg.ddpg import DDPG
from AgentModel.agent import DDPG
# from AgentModel.util import mpi_mean, mpi_std, mpi_max, mpi_sum
import torch

# from AgentModel import logger
import numpy as np
# import tensorflow as tf
# from mpi4py import MPI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAINING_STATS_SAVE_FILEPATH = '/Users/beep_kai/Desktop/fyp/CSGO_AI_ref/src/training_statistics/epoch_data/training_data'
EVALUATION_STATS_SAVE_FILEPATH = '/Users/beep_kai/Desktop/fyp/CSGO_AI_ref/src/training_statistics/epoch_data/evaluation_data/'

def train(env, nb_epochs = 40, nb_epoch_cycles = 20, nb_train_steps = 500, nb_of_rounds = 10, eval_env=None):
    print('start training')
    print('init agent and env')
    
    
    agent = DDPG(env.observation_space, env.action_space, env.goal_space, device)
    obs, p_obs, reward, done, goal, p_goal = env.reset()
    agent.reset(obs, p_obs, goal, p_goal)
    
    
    
    for epoch in range(nb_epochs):
        #init data structure here to collectn data foreach epoch
        avg_reward_in_epoch = []
        avg_winrate_in_epoch= []
        
        for epoch_cycle in range(nb_epoch_cycles):
            #init data structure here to collectn data foreach epoch cycle
            cum_reward_in_epoch_cycle = 0
            rounds_won = 0
            for round in range(nb_of_rounds):
                #init data structure here to collect data for each round
                round_winner = None
                episode_reward = 0
                episode_step = 0
                while True:
                    print("round number : ", round)
                    action, q = None, 0
                    if epoch_cycle < 1:
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
                        if obs['winner'] == 1:
                            rounds_won += 1
                        
                        obs, p_obs, reward, done, goal, p_goal = env.reset()
                        agent.reset(obs, p_obs, goal, p_goal)
                        break
            
            #collecting data
            avg_reward_in_epoch.append(cum_reward_in_epoch_cycle/nb_of_rounds)
            avg_winrate_in_epoch.append(rounds_won/nb_of_rounds)
            
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
            agent.save_model()
        np.save(f'{TRAINING_STATS_SAVE_FILEPATH}epoch_{epoch}_avg_reward.npy', np.array(avg_reward_in_epoch))
        np.save(f'{TRAINING_STATS_SAVE_FILEPATH}epoch_{epoch}_avg_winrate.npy', np.array(avg_winrate_in_epoch))    

def train1(env, nb_epochs = 40, nb_epoch_cycles = 20, nb_train_steps = 500, nb_of_rounds = 10, eval_env=None):
    # rank = MPI.COMM_WORLD.Get_rank()

    print("start training")

    agent = DDPG(env.observation_space, env.action_space, env.goal_space, device)


    
    # step = 0
    # episode = 0
    # eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)


    #init agent and env
    obs, p_obs, reward, done, goal, p_goal = env.reset()
    agent.reset(obs, p_obs, goal, p_goal)


    # if eval_env is not None:
    #     eval_obs = eval_env.reset()

    done = False
    episode_reward = 0.
    episode_step = 0
    episodes = 0
    t = 0

    epoch = 0
    start_time = time.time()

    epoch_episode_rewards = []
    epoch_episode_steps = []
    epoch_start_time = time.time()
    epoch_actions = []
    epoch_qs = []
    epoch_episodes = 0
    print('Start training')
    for epoch in range(nb_epochs):
        for cycle in range(nb_epoch_cycles):
            # Perform rollouts.
            for round in range(nb_of_rounds):
                while True:
                    print("round number : ", round)
                    # Predict next action.
                    action, q = None, 0
                    if cycle < 1:
                        action, q = agent.random_action(), 0
                    else:
                        action, q = agent.select_action(p_obs, p_goal), 0
                    obs = env.get_current_observation()
                    new_obs, new_p_obs, r, done, info = env.step(action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    goal, p_goal = info['goal state'], info['partial goal state']
                    new_obs = env.get_current_observation()

                    t += 1
                    episode_reward += r
                    episode_step += 1

                    # Book-keeping.
                    epoch_actions.append(action)
                    epoch_qs.append(q)
                    agent.observe(obs, p_obs, action, r, new_obs, new_p_obs, goal, p_goal ,done)
                    p_obs = new_p_obs
                    print("done: ", done)
                
                    if done:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        episode_reward = 0
                        episode_step = 0
                        epoch_episodes += 1
                        episodes += 1

                        obs, p_obs, reward, done, goal, p_goal = env.reset()
                        agent.reset(obs, p_obs, goal, p_goal)
                        break
            # Train.
            epoch_actor_losses = []
            epoch_critic_losses = []
            for _ in range(nb_train_steps):
                cl, al = agent.update_policy()
                epoch_critic_losses.append(cl)
                epoch_actor_losses.append(al)



        if epoch % 4 == 0:
            agent.save_model()
        # Log stats.
        epoch_train_duration = time.time() - epoch_start_time
        print("epoch number: ", epoch)
        print('Epoch duration: {:.2f}s'.format(epoch_train_duration))
        
        duration = time.time() - start_time
        stats = agent.get_stats()
        combined_stats = {}
        for key in sorted(stats.keys()):
            combined_stats[key] = mpi_mean(stats[key])

        # Rollout statistics.
        combined_stats['rollout/return'] = mpi_mean(epoch_episode_rewards)
        combined_stats['rollout/return_history'] = mpi_mean(np.mean(episode_rewards_history))
        combined_stats['rollout/episode_steps'] = mpi_mean(epoch_episode_steps)
        combined_stats['rollout/episodes'] = mpi_sum(epoch_episodes)
        combined_stats['rollout/actions_mean'] = mpi_mean(epoch_actions)
        combined_stats['rollout/actions_std'] = mpi_std(epoch_actions)
        combined_stats['rollout/Q_mean'] = mpi_mean(epoch_qs)

        # Train statistics.
        combined_stats['train/loss_actor'] = mpi_mean(epoch_actor_losses)
        combined_stats['train/loss_critic'] = mpi_mean(epoch_critic_losses)


        # Total statistics.
        combined_stats['total/duration'] = mpi_mean(duration)
        combined_stats['total/steps_per_second'] = mpi_mean(float(t) / float(duration))
        combined_stats['total/episodes'] = mpi_mean(episodes)
        combined_stats['total/epochs'] = epoch + 1
        combined_stats['total/steps'] = t
    
    df = pd.DataFrame.from_dict(combined_stats, orient='index')
    df.to_csv('stats.csv')
  


def train2(env, nb_epochs = 40, nb_epoch_cycles = 20, nb_train_steps = 50, nb_rollout_steps = 500, nb_eval_steps = 100, batch_size = 128, eval_env=None):
    # rank = MPI.COMM_WORLD.Get_rank()

    print("start training")
    # assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    
    # max_action = env.action_space.high
    # logger.info('scaling actions by {} before executing in env'.format(max_action))
    
    # agent = DDPG(env.state_space.shape, env.observation_space.shape, env.action_space.shape)
    agent = DDPG(env.observation_space, env.action_space, env.goal_space, device)


    # logger.info('Using agent with the following configuration:')
    # logger.info(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.
    # if rank == 0:
    #     saver = tf.train.Saver()
    # else:
    #     saver = None
    
    step = 0
    episode = 0
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    # with U.single_threaded_session() as sess:
    # Prepare everything.
    # agent.initialize(sess)
    # sess.graph.finalize()

    #init agent and env
    obs, p_obs, reward, done, goal, p_goal = env.reset()
    agent.reset(obs, p_obs, goal, p_goal)


    if eval_env is not None:
        eval_obs = eval_env.reset()

    done = False
    episode_reward = 0.
    episode_step = 0
    episodes = 0
    t = 0

    epoch = 0
    start_time = time.time()

    epoch_episode_rewards = []
    epoch_episode_steps = []
    epoch_episode_eval_rewards = []
    epoch_episode_eval_steps = []
    epoch_start_time = time.time()
    epoch_actions = []
    epoch_qs = []
    epoch_episodes = 0
    print('Start training')
    for epoch in range(nb_epochs):
        for cycle in range(nb_epoch_cycles):
            # Perform rollouts.
            for t_rollout in range(nb_rollout_steps):
                print("t_rollout: ", t_rollout)
                
                # Predict next action.
                action, q = None, 0
                if cycle < 1:
                    action, q = agent.random_action(), 0
                else:
                    action, q = agent.select_action(p_obs, p_goal), 0
                obs = env.get_current_observation()
                # assert action.shape == env.action_space.shape
                # assert max_action.shape == action.shape
                new_obs, new_p_obs, r, done, info = env.step(action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                goal, p_goal = info['goal state'], info['partial goal state']
                new_obs = env.get_current_observation()

                t += 1
                episode_reward += r
                episode_step += 1

                # Book-keeping.
                epoch_actions.append(action)
                epoch_qs.append(q)
                agent.observe(obs, p_obs, action, r, new_obs, new_p_obs, goal, p_goal ,done)
                p_obs = new_p_obs
                print("done: ", done)
                
                if done:
                    # Episode done.
                    epoch_episode_rewards.append(episode_reward)
                    episode_rewards_history.append(episode_reward)
                    epoch_episode_steps.append(episode_step)
                    episode_reward = 0
                    episode_step = 0
                    epoch_episodes += 1
                    episodes += 1

                    agent.reset(obs, p_obs, goal, p_goal)
                    obs, p_obs, reward, done, goal, p_goal = env.reset()
                    break

            # Train.
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_adaptive_distances = []
            for t_train in range(nb_train_steps):

                # cl, al = agent.train()
                cl, al = agent.update_policy()
                epoch_critic_losses.append(cl)
                epoch_actor_losses.append(al)
                # agent.update_target_net()


            # Evaluate for 1 round
            # eval_episode_rewards = []
            # eval_qs = []
            # if eval_env is not None:
            #     eval_episode_reward = 0.
            #     for t_rollout in range(nb_eval_steps):
            #         eval_action, eval_q = agent.pi(eval_obs, p_goal, apply_noise=False, compute_Q=True)
            #         eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
            #         eval_episode_reward += eval_r

            #         eval_qs.append(eval_q)
            #         if eval_done:
            #             eval_obs = eval_env.reset()
            #             eval_episode_rewards.append(eval_episode_reward)
            #             eval_episode_rewards_history.append(eval_episode_reward)
            #             eval_episode_reward = 0.

        if epoch % 4 == 0:
            agent.save_model()
        # Log stats.
        epoch_train_duration = time.time() - epoch_start_time
        duration = time.time() - start_time
        stats = agent.get_stats()
        combined_stats = {}
        for key in sorted(stats.keys()):
            combined_stats[key] = mpi_mean(stats[key])

        # Rollout statistics.
        combined_stats['rollout/return'] = mpi_mean(epoch_episode_rewards)
        combined_stats['rollout/return_history'] = mpi_mean(np.mean(episode_rewards_history))
        combined_stats['rollout/episode_steps'] = mpi_mean(epoch_episode_steps)
        combined_stats['rollout/episodes'] = mpi_sum(epoch_episodes)
        combined_stats['rollout/actions_mean'] = mpi_mean(epoch_actions)
        combined_stats['rollout/actions_std'] = mpi_std(epoch_actions)
        combined_stats['rollout/Q_mean'] = mpi_mean(epoch_qs)

        # Train statistics.
        combined_stats['train/loss_actor'] = mpi_mean(epoch_actor_losses)
        combined_stats['train/loss_critic'] = mpi_mean(epoch_critic_losses)
        combined_stats['train/param_noise_distance'] = mpi_mean(epoch_adaptive_distances)

        # Evaluation statistics.
        # if eval_env is not None:
        #     combined_stats['eval/return'] = mpi_mean(eval_episode_rewards)
        #     combined_stats['eval/return_history'] = mpi_mean(np.mean(eval_episode_rewards_history))
        #     combined_stats['eval/Q'] = mpi_mean(eval_qs)
        #     combined_stats['eval/episodes'] = mpi_mean(len(eval_episode_rewards))

        # Total statistics.
        combined_stats['total/duration'] = mpi_mean(duration)
        combined_stats['total/steps_per_second'] = mpi_mean(float(t) / float(duration))
        combined_stats['total/episodes'] = mpi_mean(episodes)
        combined_stats['total/epochs'] = epoch + 1
        combined_stats['total/steps'] = t
        
        # for key in sorted(combined_stats.keys()):
        #     logger.record_tabular(key, combined_stats[key])
        # logger.dump_tabular()
        # logger.info('')
        # logdir = logger.get_dir()
        # if rank == 0 and logdir:
        #     if hasattr(env, 'get_state'):
        #         with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
        #             pickle.dump(env.get_state(), f)
        #     if eval_env and hasattr(eval_env, 'get_state'):
        #         with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
        #             pickle.dump(eval_env.get_state(), f)
