import numpy as np
from stable_baselines3 import PPO
from racing_env import RacingEnv
from simulate_pursuit import simulate_track_pursuit
from simulate_raceline import simulate_raceline
from track import load_track
import matplotlib.pyplot as plt

from visualisation import plot_multiple_trajectories


if __name__ == "__main__":
    # env = RacingEnv("Austin")

    # # Train
    # model = PPO("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=50000)
    # model.save("ppo_racing_austin")

    
    # obs = env.reset()
    # done = False
    # while not done:
    #     # get action from model
    #     action, _ = model.predict(obs, deterministic=True)
    #     # take action in environment
    #     obs, _, done, info = env.step(action)
    #     print(info)
        
    # env.render()
    track = load_track("Austin")
    time_raceline = simulate_raceline(track, plot_speed=True)
    print("Ideal lap time on raceline: {:.2f} seconds".format(time_raceline))

    # Simulate pursuit
    time_pursuit = simulate_track_pursuit(track, plot_speed=True)
    print("Lap time on pursuit: {:.2f} seconds".format(time_pursuit))
