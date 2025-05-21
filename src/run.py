import numpy as np
from stable_baselines3 import TD3
from racing_env import RacingEnv
from simulate_pursuit import simulate_track_pursuit
from simulate_raceline import simulate_raceline
from track import load_track
import matplotlib.pyplot as plt
from visualisation import plot_multiple_trajectories
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

if __name__ == "__main__":
    env = RacingEnv("Spielberg", dt=0.1, reward_factor=0.5)

    # Train
    model = TD3("MlpPolicy", env, learning_rate=3e-4, verbose=1, )
    model.learn(total_timesteps=20000, progress_bar=True)
    model.save("td3_racing_spielberg")

    # Test
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
    # render the environment
    env.render()
    print(info)
