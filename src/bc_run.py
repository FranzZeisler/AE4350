import pickle
import numpy as np
from stable_baselines3 import TD3
import torch
from behavior_cloning import train_bc
from racing_env import RacingEnv
from simulate_pursuit import simulate_track_pursuit
from track import load_track
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

if __name__ == "__main__":
    # Name of the track to simulate
    track_name = "Spielberg"
    track = load_track(track_name)
    #---------------------------------------------------------------------------------------------------
    # Step 1: Simulate the track using pure pursuit control to generate expert dataset
    #---------------------------------------------------------------------------------------------------
    print("Step 1 - Simulating track with pure pursuit control to generate expert dataset")

    # Simulate the car on the track using pure pursuit control
    time_elapsed, expert_dataset = simulate_track_pursuit(track)

    # Print the time elapsed during the simulation
    print(f"Time elapsed: {time_elapsed:.2f} seconds")

    # Save the expert dataset to a file as trackname_expert_dataset.npy
    expert_dataset_path = f"{track_name}_expert_dataset.pkl"
    with open(expert_dataset_path, "wb") as f:
        pickle.dump(expert_dataset, f)

    print(f"Expert dataset saved as {expert_dataset_path}")
    #---------------------------------------------------------------------------------------------------
    # Step 2: Train Behavior Cloning model on expert dataset
    #---------------------------------------------------------------------------------------------------
    print("Step 2 - Training Behavior Cloning model on expert dataset")
    print("Starting Behavior Cloning training...")

    # Load the expert dataset from pickle file
    with open(f"{track_name}_expert_dataset.pkl", "rb") as f:
        expert_dataset = pickle.load(f)

    # Train the Behavior Cloning model
    bc_model = train_bc(expert_dataset, epochs=100)
    print("Behavior Cloning training completed.")

    # Save the trained BC model weights
    bc_weights_path = f"bc_actor{track_name}.pth"
    torch.save(bc_model.state_dict(), bc_weights_path)
    print(f"BC model weights saved as {bc_weights_path}")
    #---------------------------------------------------------------------------------------------------
    # Step 3: Train TD3 WITH BC warm start
    #---------------------------------------------------------------------------------------------------
    print("Step 3 - Training TD3 WITH Behavior Cloning warm-up")
    env = RacingEnv(track_name=track_name, dt=0.1)
    model_bc = TD3("MlpPolicy", env, learning_rate=3e-4, verbose=1)

    input_dim = expert_dataset[0][0].shape[0]
    output_dim = expert_dataset[0][1].shape[0]
    from behavior_cloning import BCActor
    bc_actor = BCActor(input_dim, output_dim)
    bc_actor.load_state_dict(torch.load(bc_weights_path))

    # Copy BC weights into TD3 actor
    with torch.no_grad():
        for bc_param, td3_param in zip(bc_actor.parameters(), model_bc.policy.actor.parameters()):
            td3_param.copy_(bc_param)

    print("Loaded BC weights into TD3 actor manually.")
    model_bc.learn(total_timesteps=50000, progress_bar=True)
    model_bc.save(f"td3_{track_name}_with_bc.zip")
    print(f"TD3 model with BC warm start saved.")
    #---------------------------------------------------------------------------------------------------
    # Step 4 Evaluate the trained TD3 model with BC warm start
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

