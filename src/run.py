import numpy as np
from stable_baselines3 import PPO
from racing_env import RacingEnv
from simulate_pursuit import simulate_track_pursuit
from track import load_track
import matplotlib.pyplot as plt

from visualisation import plot_multiple_trajectories


if __name__ == "__main__":
    env = RacingEnv("Austin")

    # Train
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save("ppo_racing_austin")

    
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        # after episode, store trajectory and crash point
        
    env.render()
