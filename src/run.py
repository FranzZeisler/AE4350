import numpy as np
from stable_baselines3 import PPO
from racing_env import RacingEnv
from simulate_pursuit import simulate_track_pursuit
from simulate_raceline import simulate_raceline
from track import load_track
import matplotlib.pyplot as plt
from visualisation import plot_multiple_trajectories
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

if __name__ == "__main__":
    env = RacingEnv("Spa", dt=0.1, acceleration_reward=500)

    # Train
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=200000)
    model.save("ppo_racing_austin")

    # Test
    obs = env.reset()
    done = False
    while not done:
        # get action from model
        action, _ = model.predict(obs, deterministic=True)
        # take action in environment
        obs, _, done, info = env.step(action)
    
    # render the environment
    env.render()
    print(info)
        
    # track = load_track("YasMarina")
    # # Simulate pursuit
    # time_pursuit = simulate_track_pursuit(track, plot_speed=True)
    # minutes = int(time_pursuit // 60)
    # seconds = int(time_pursuit % 60)
    # milliseconds = int((time_pursuit - int(time_pursuit)) * 1000)

    # print(f"Lap time on pursuit: {minutes}:{seconds:02d}.{milliseconds:03d}")
