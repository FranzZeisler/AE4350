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
    model_bc = TD3("MlpPolicy", env, learning_rate=3e-3, verbose=1)

    model_bc.learn(total_timesteps=50000, progress_bar=True)

    #---------------------------------------------------------------------------------------------------
    print("Step 4 - Evaluating the trained TD3 model with BC warm start")
    
    num_eval_episodes = 5
    lap_times = []
    total_rewards = []

    for episode in range(num_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            action, _ = model_bc.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        
        if "lap_time" in info:
            lap_times.append(info["lap_time"])
            print(f"Episode {episode+1} lap time: {info['lap_time']}")
        else:
            print(f"Episode {episode+1} finished without lap completion.")

        env.render()
        
    print(f"\nAverage lap time over {num_eval_episodes} episodes: {np.mean(lap_times) if lap_times else 'N/A'}")
    print(f"Average total reward over {num_eval_episodes} episodes: {np.mean(total_rewards)}")

