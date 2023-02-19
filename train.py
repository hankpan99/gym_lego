import os
import gym
import time
import torch
import argparse
import shutil
import numpy as np
# from lego_env import LegoEnv
from dexycb_env import DexYCBEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


def parse_args():
    parser = argparse.ArgumentParser(description='Training settings')
    parser.add_argument('--GUI', action='store_true')
    parser.add_argument('--method', type=str, default="")
    parser.add_argument('--n_envs', type=int, default=32)

    parser.set_defaults(GUI=False)

    args = parser.parse_args()

    return args

def create_ckpt_dirs(args, filename_list):
    timestamp = int(time.time())

    models_dir = f"models/{timestamp}_{args.method}/"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    log_dir = f"models/{timestamp}_{args.method}/logs/"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # backup files
    for filename in filename_list:
        shutil.copyfile(filename, f"models/{timestamp}_{args.method}/{filename}")
    
    return models_dir, log_dir

def tensorboard_launcher(directory_path):
    from tensorboard import program
    import webbrowser
    # learning visualizer
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', directory_path])
    url = tb.launch()
    print("Tensorboard session created: " + url)
    webbrowser.open_new(url)

### parse arguments
args = parse_args()
num_envs = args.n_envs

### setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

### Set training step parameters
pre_grasp_steps = 60
trail_steps = 135

grasp_steps = pre_grasp_steps
n_steps = grasp_steps  + trail_steps
total_steps = n_steps * num_envs

# env = make_vec_env(LegoEnv, n_envs=args.n_envs, env_kwargs={"args":args}, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
random_noise_pos = np.random.uniform([-0.02, -0.02, 0],[0.02, 0.02, 0], 3).copy()
random_noise_qpos = np.random.uniform(-0.05, 0.05, 48).copy()
env = make_vec_env(DexYCBEnv,
                   n_envs=num_envs,
                   env_kwargs={"args":args, 'random_noise':[random_noise_pos, random_noise_qpos], 'max_steps':n_steps},
                   vec_env_cls=SubprocVecEnv,
                   vec_env_kwargs=dict(start_method='fork'))
env.reset()

### Set up logging
models_dir, log_dir = create_ckpt_dirs(args, ['dexycb_env.py', 'dexycb_data_all.pickle', __file__])
tensorboard_launcher(log_dir)

### Set up RL algorithm
model = PPO('MlpPolicy',
            env,
            n_steps=n_steps,
            batch_size=4,
            verbose=1,
            tensorboard_log=log_dir,
            device=device)
iters = 0

while True:
    model.learn(total_timesteps=n_steps * num_envs,
                reset_num_timesteps=False)
    
    if iters % 10 == 0:
        model.save(f"{models_dir}/{iters}")
    
    iters += 1