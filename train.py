import gymnasium as gym
from stable_baselines3 import PPO
from rl_models.ppo import PPO as myPPO
from rl_models.reinforce import Reinforce
import numpy as np


ENV_ID = "CartPole-v1"
#ENV_ID = "MountainCar-v0"
#ENV_ID = "Acrobot-v1"
env = gym.make(ENV_ID)

# create the model

#model = PPO(policy="MlpPolicy", env=env, verbose=1)
#model = Reinforce(env=env)
model = myPPO(env=env)



# train the model
model.learn(total_timesteps=204_800)


# run the model
watch_env = gym.make(ENV_ID, render_mode="human")

obs, info = watch_env.reset()
episode_return = 0.0

for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    next_obs, reward, terminated, truncated, info = watch_env.step(action)
    episode_return += reward

    # Optional: print each step’s diagnostics (comment out if too noisy)
    # print(f"Step {step}: obs={obs}, action={action}, reward={reward}")

    if terminated or truncated:
        print("\nEpisode ended!")
        print("return", episode_return)
        
        # Print termination reason
        if terminated:
            print("Reason: TERMINATED")
        elif truncated:
            print("Reason: TRUNCATED")
        break

    obs = next_obs

watch_env.close()
