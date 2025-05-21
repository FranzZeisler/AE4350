import numpy as np
from stable_baselines3 import TD3
import torch
from behavior_cloning import train_bc
from racing_env import RacingEnv
from simulate_pursuit import simulate_track_pursuit
from simulate_raceline import simulate_raceline
from track import load_track
import matplotlib.pyplot as plt
from visualisation import plot_multiple_trajectories
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
    np.save(track_name + "_expert_dataset.npy", expert_dataset)
    print("Expert dataset saved as " + track_name + "_expert_dataset.npy")

    #---------------------------------------------------------------------------------------------------
    # Step 2: Train Behavior Cloning model on expert dataset
    #---------------------------------------------------------------------------------------------------
    print("Step 2 - Training Behavior Cloning model on expert dataset")
    # Train Behavior Cloning model on expert dataset
    print("Starting Behavior Cloning training...")
    bc_model = train_bc(expert_dataset, epochs=100)
    print("Behavior Cloning training completed.")
    # Optionally save the trained BC model weights as bc_actor_trackname.pth
    bc_weights_path = "bc_actor" + track_name + ".pth"
    torch.save(bc_model.state_dict(), bc_weights_path)
    print(f"BC model weights saved as {bc_weights_path}")

    #---------------------------------------------------------------------------------------------------
    # Step 3: Train TD3 with Behavior Cloning warm-up
    #---------------------------------------------------------------------------------------------------
    print("Step 3 - Training TD3 with Behavior Cloning warm-up")
    env = RacingEnv(track_name, dt=0.1)
    # Create TD3 model
    model = TD3("MlpPolicy", env, learning_rate=3e-4, verbose=1)

    # Load BC weights
    bc_weights_path = "bc_actor" + track_name + ".pth"
    bc_state_dict = torch.load(bc_weights_path)
    model.policy.actor.load_state_dict(bc_state_dict)
    print("Loaded BC weights into TD3 actor network")


    # Continue training with TD3
    total_timesteps = 100000  # e.g., increase to desired amount
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    # Save the improved model
    model_path = "td3_" + track_name + ".zip"
    model.save(model_path)
    print(f"TD3 model saved as {model_path}")
    #---------------------------------------------------------------------------------------------------
    # Step 4: Evaluate the trained TD3 model
    #---------------------------------------------------------------------------------------------------
    print("Step 4 - Evaluating the trained TD3 model")
    
    # Reset the environment
    obs = env.reset()
    done = False
  
    # Loop until the episode is done
    while not done:
        # Get action from the TD3 model
        action, _ = model.predict(obs, deterministic=True)
        # Step the environment
        obs, reward, done, info = env.step(action)
        # Append position and speed to lists
    
    # Print the lap time and render the trajectory
    print(info)
    env.render()
