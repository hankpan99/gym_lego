from stable_baselines3 import PPO
from lego_env import LegoEnv

# with (open("dgrasp_data.pickle", "rb")) as openfile:
#     action1=pickle.load(openfile)["subgoal_1"]["hand_traj_reach"].reshape((38,51))
# with (open("dgrasp_data.pickle", "rb")) as openfile:
#     action2=pickle.load(openfile)["subgoal_1"]["hand_ref_pose"].reshape((1,51))
# with (open("dgrasp_data.pickle", "rb")) as openfile:
#     action3=pickle.load(openfile)["subgoal_1"]["hand_traj_grasp"].reshape((28,51))
# action=np.vstack([action1, action2, action3])

models_dir = "models/1671952788"

env=LegoEnv()
env.reset()

model_path = f"{models_dir}/290000.zip"
model = PPO.load(model_path, env=env)

episodes = 5

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        print(rewards)