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
from helper.raisim_gym_helper import ConfigurationSaver, load_param
import torch.nn as nn
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Training settings')
    parser.add_argument('--GUI', action='store_true')
    parser.add_argument('--weight', type=str, default='')

    parser.set_defaults(GUI=False)

    args = parser.parse_args()

    return args

### parse arguments
args = parse_args()

weight_path = args.weight

### setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

env = DexYCBEnv(args, [np.zeros(3), np.zeros(48)])

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

### Set up logging
saver = ConfigurationSaver(log_dir = "./models", save_items=[], test_dir=True)

### Set up RL algorithm
actor = ppo_module.Actor(ppo_module.MLP([128, 128], output_activation, activations, ob_dim, act_dim, False),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, 1.0),device)

critic = ppo_module.Critic(ppo_module.MLP([128, 128], output_activation, activations, ob_dim, 1, False),device)

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=1,
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.996,
              lam=0.95,
              num_mini_batches=4,
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False
              )

print(saver.data_dir)
load_param(saver.data_dir.split('eval')[0] + weight_path, env, actor, critic, ppo.optimizer, saver.data_dir, None, store_again=False)

episodes = 100

for ep in range(episodes):
    obs = env.reset().astype('float64')

    for step in range(n_steps):
        ### Get action from policy
        action_pred = actor.architecture.architecture(torch.from_numpy(obs.astype('float64')).float().to(device))
        frame_start = time.time()

        action_ll = action_pred.cpu().detach().numpy()

        ### After grasp is established (set to motion synthesis mode)
        if step > grasp_steps:
            env.set_root_control()

        obs, rewards, done, info = env.step(action_ll.astype('float64'))