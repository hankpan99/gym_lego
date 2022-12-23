from stable_baselines3.common.env_checker import check_env
from lego_env import LegoEnv

env=LegoEnv()
check_env(env)

episodes = 3

for episode in range(episodes):
	done = False
	obs = env.reset()
	while True:#not done:
		random_action = env.action_space.sample()
		print("action",random_action)
		obs, reward, done, info = env.step(random_action)
		print('reward',reward)