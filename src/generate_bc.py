import os
import pickle
import torch
from behavior_cloning import train_bc, BCActor
from racing_env import RacingEnv
from simulate_pursuit import simulate_track_pursuit
from track import load_track

if __name__ == "__main__":
    # List of tracks to simulate
    track_names = ["Austin", "Budapest", "Melbourne", "MexicoCity", "Monza", 
                   "SaoPaulo", "Spa", "Spielberg", "Suzuka", "Zandvoort"]

    # Create directory to save the expert datasets and models
    save_dir = "expert_datasets_and_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # Create the directory if it doesn't exist

    # Loop over each track
    for track_name in track_names:
        print(f"\nProcessing track: {track_name}")

        # Load the track
        track = load_track(track_name)

        #---------------------------------------------------------------------------------------------------
        # Step 1: Simulate the track using pure pursuit control to generate expert dataset
        #---------------------------------------------------------------------------------------------------
        print("Step 1 - Simulating track with pure pursuit control to generate expert dataset")

        # Simulate the car on the track using pure pursuit control
        time_elapsed, expert_dataset = simulate_track_pursuit(track)

        # Print the time elapsed during the simulation
        print(f"Time elapsed: {time_elapsed:.2f} seconds")

        # Save the expert dataset to the expert_datasets_and_models folder
        expert_dataset_path = os.path.join(save_dir, f"{track_name}_expert_dataset.pkl")
        with open(expert_dataset_path, "wb") as f:
            pickle.dump(expert_dataset, f)

        print(f"Expert dataset saved as {expert_dataset_path}")

        #---------------------------------------------------------------------------------------------------
        # Step 2: Train Behavior Cloning model on expert dataset
        #---------------------------------------------------------------------------------------------------
        print("Step 2 - Training Behavior Cloning model on expert dataset")
        print("Starting Behavior Cloning training...")

        # Load the expert dataset from pickle file
        with open(expert_dataset_path, "rb") as f:
            expert_dataset = pickle.load(f)

        # Train the Behavior Cloning model
        bc_model = train_bc(expert_dataset, epochs=100)
        print("Behavior Cloning training completed.")

        # Save the trained BC model weights to the expert_datasets_and_models folder
        bc_weights_path = os.path.join(save_dir, f"bc_actor_{track_name}.pth")
        torch.save(bc_model.state_dict(), bc_weights_path)
        print(f"BC model weights saved as {bc_weights_path}")

        #---------------------------------------------------------------------------------------------------
        # Step 2.5: Evaluate BC Alone before training TD3
        #---------------------------------------------------------------------------------------------------
        print("Step 2.5 - Evaluating Behavior Cloning (BC) policy alone")

        # Create the environment
        env = RacingEnv(track_name=track_name, dt=0.1, discount_factor=0.9775)

        # Load the BC model
        input_dim = expert_dataset[0][0].shape[0]  # Assuming expert dataset is in format (state, action)
        output_dim = expert_dataset[0][1].shape[0]  # (state, action)
        bc_actor = BCActor(input_dim, output_dim)
        bc_actor.load_state_dict(torch.load(bc_weights_path))
        bc_actor.eval()

        # Reset the environment
        obs = env.reset()
        done = False

        # Loop until the episode is done
        while not done:
            # Get action from the BC model
            action = bc_actor(torch.FloatTensor(obs)).detach().numpy()

            # Step the environment
            obs, reward, done, info = env.step(action)

        # Print the lap time and render the trajectory
        print(info)
        env.render()
