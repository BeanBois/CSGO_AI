import os
import time
from collections import deque
import pickle

# from baselines.ddpg.ddpg import DDPG
from AgentModel.agent import DDPG
# from AgentModel.util import mpi_mean, mpi_std, mpi_max, mpi_sum
import torch

# from AgentModel import logger
import numpy as np
# import tensorflow as tf
# from mpi4py import MPI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(env, nb_epochs = 40, nb_epoch_cycles = 20, nb_train_steps = 50, nb_rollout_steps = 500, nb_eval_steps = 100, batch_size = 128, eval_env=None):
    # rank = MPI.COMM_WORLD.Get_rank()

    print("start training")
    # assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    
    # max_action = env.action_space.high
    # logger.info('scaling actions by {} before executing in env'.format(max_action))
    
    # agent = DDPG(env.state_space.shape, env.observation_space.shape, env.action_space.shape)
    
    agent = DDPG(state_space = env.observation_space_size, \
            action_space = env.action_space_size, \
            goal_space = env.goal_space_size, \
            device = device,\
            state_dim = env.observation_space,\
            action_dim = env.action_space,\
            goal_dim= env.goal_space,\
                                )


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
                action, q = agent.select_action(p_obs, p_goal), None
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
                    episode_reward = 0.
                    episode_step = 0
                    epoch_episodes += 1
                    episodes += 1

                    agent.reset(obs, p_obs, goal, p_goal)
                    obs, p_obs, reward, done, goal, p_goal = env.reset()

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
