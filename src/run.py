import warnings

import numpy as np
from stable_baselines3 import TD3

from racing_env import RacingEnv
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

if __name__ == "__main__":
    # Name of the track to simulate
    track_name = "Spielberg"
    
    env = RacingEnv(track_name=track_name, dt=0.1)
    env.seed(42)
    model_bc = TD3("MlpPolicy", env, learning_rate=3e-4, verbose=1)

    model_bc.learn(total_timesteps=50000, progress_bar=True)

    #---------------------------------------------------------------------------------------------------
    print("Step 4 - Evaluating the trained TD3 model with BC warm start")
    obs = env.reset()
    done = False
    episode_reward = 0.0
    
    while not done:
        action, _ = model_bc.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
    
    print(info)
    env.render()
        
    
