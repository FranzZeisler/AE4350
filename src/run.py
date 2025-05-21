import warnings

from stable_baselines3 import TD3

from racing_env import RacingEnv
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

if __name__ == "__main__":
    # Name of the track to simulate
    track_name = "Spielberg"
    
    env = RacingEnv(track_name=track_name, dt=0.1)
    model_bc = TD3("MlpPolicy", env, learning_rate=3e-4, verbose=1)

    model_bc.learn(total_timesteps=20000, progress_bar=True)

    #---------------------------------------------------------------------------------------------------
    print("Step 4 - Evaluating the trained TD3 model with BC warm start")
    
    # Reset the environment
    obs = env.reset()
    done = False
  
    # Loop until the episode is done
    while not done:
        # Get action from the TD3 model
        action, _ = model_bc.predict(obs, deterministic=True)
        # Step the environment
        obs, reward, done, info = env.step(action)
        # Append position and speed to lists
    
    # Print the lap time and render the trajectory
    print(info)
    env.render()

