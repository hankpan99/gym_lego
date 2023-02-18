import os
import gym
import time
import torch
import argparse
import shutil
# from lego_env import LegoEnv
from dexycb_env import DexYCBEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import algo.ppo.module as ppo_module
import algo.ppo.ppo as PPO
from helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
import torch.nn as nn
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Training settings')
    parser.add_argument('--GUI', action='store_true')
    parser.add_argument('--method', type=str, default="")
    parser.add_argument('--n_envs', type=int, default=32)

    parser.set_defaults(GUI=False)

    args = parser.parse_args()

    return args

### parse arguments
args = parse_args()

### setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_envs = args.n_envs
random_noise_pos = np.random.uniform([-0.02, -0.02, 0],[0.02, 0.02, 0], 3).copy()
random_noise_qpos = np.random.uniform(-0.05, 0.05, 48).copy()
env = make_vec_env(DexYCBEnv,
                   n_envs=args.n_envs,
                   env_kwargs={"args":args, 'random_noise':[random_noise_pos, random_noise_qpos]},
                   vec_env_cls=SubprocVecEnv,
                   vec_env_kwargs=dict(start_method='fork'))

### get network parameters
activations = nn.LeakyReLU
output_activation = nn.Tanh
ob_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

### Set training step parameters
pre_grasp_steps = 60
trail_steps = 135

grasp_steps = pre_grasp_steps
n_steps = grasp_steps  + trail_steps
total_steps = n_steps * env.num_envs

### Set up logging
saver = ConfigurationSaver(log_dir = "./models",
                           save_items=["./dexycb_data_all.pickle", "./dexycb_env.py", "./runner.py"])
tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update

### Set up RL algorithm
actor = ppo_module.Actor(ppo_module.MLP([128, 128], output_activation, activations, ob_dim, act_dim, False),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, 1.0),device)

critic = ppo_module.Critic(ppo_module.MLP([128, 128], output_activation, activations, ob_dim, 1, False),device)

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=num_envs,
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.996,
              lam=0.95,
              num_mini_batches=4,
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False
              )

avg_rewards = []
for update in range(3001):
    start = time.time()

    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.

    ### Store policy
    if update % 200 == 0:
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')

        # env.save_scaling(saver.data_dir, str(update))

    ### Run episode rollouts
    obs = env.reset()
    for step in range(n_steps):
        # obs = env.observe().astype('float64')

        action = ppo.observe(obs)
        # reward, dones = env.step(action.astype('float64'))
        obs, reward, dones, info = env.step(action.astype('float64'))
        reward.clip(min=-2)

        ppo.step(value_obs=obs, rews=reward, dones=np.zeros(num_envs, dtype=bool))
        done_sum = done_sum + np.sum(np.zeros(num_envs, dtype=bool))
        reward_ll_sum = reward_ll_sum + np.sum(reward)
    # obs = env.observe().astype('float64')

    ### Update policy
    ppo.update(actor_obs=obs, value_obs=obs, log_this_iteration=update % 10 == 0, update=update)
    average_ll_performance = reward_ll_sum / total_steps
    average_dones = done_sum / total_steps
    avg_rewards.append(average_ll_performance)

    actor.distribution.enforce_minimum_std((torch.ones(act_dim)*0.2).to(device))

    end = time.time()

    ### Log results
    mean_file_name = saver.data_dir + "/rewards.txt"
    np.savetxt(mean_file_name, avg_rewards)

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    # print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
    #                                                                    * cfg['environment']['control_dt'])))
    print('std: ')
    print(np.exp(actor.distribution.std.cpu().detach().numpy()))
    print('----------------------------------------------------\n')

