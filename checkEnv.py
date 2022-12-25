from stable_baselines3.common.env_checker import check_env
from lego_env import LegoEnv
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training settings')
    parser.add_argument('--GUI', action='store_true')

    parser.set_defaults(GUI=False)

    args = parser.parse_args()

    return args

args = parse_args()

env=LegoEnv(args)
check_env(env)

episodes = 1

for episode in range(episodes):
	done = False
	obs = env.reset()
	while not done:
		random_action = env.action_space.sample()
		# print("action",random_action)
		obs, reward, done, info = env.step(random_action)
		# print('reward',reward)