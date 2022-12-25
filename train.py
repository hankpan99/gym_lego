import os
import time
import torch
from lego_env import LegoEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using device', device)

env = make_vec_env(LegoEnv, n_envs=4)
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir, device=device)

TIMESTEPS = 10000
iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")