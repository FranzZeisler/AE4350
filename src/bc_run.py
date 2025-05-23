import os
import pickle
import numpy as np
import torch
import logging
import warnings
from stable_baselines3 import TD3
from behavior_cloning import train_bc, BCActor
from racing_env import RacingEnv
from simulate_pursuit import simulate_track_pursuit
from track import load_track
from visualisation import plot_track_and_trajectory

# === Configurable Parameters ===
TRACK_NAME = "Spielberg"
DISCOUNT_FACTOR = 0.98
BC_EPOCHS = 300
TD3_TIMESTEPS = 30000
BATCH_SIZE = 64
BC_WEIGHTS_PATH = f"bc_actor_{TRACK_NAME}.pth"
EXPERT_DATASET_PATH = f"{TRACK_NAME}_expert_dataset.pkl"
TD3_MODEL_PATH = f"td3_{TRACK_NAME}_with_bc.zip"
BEST_LAP_LOG = "log_best_lap_time.pkl"

# === Logging Setup ===
logging.basicConfig(filename='log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

def main(skip_sim=False, skip_bc=False):
    logging.info(f"Loading track: {TRACK_NAME}")
    track = load_track(TRACK_NAME)

    # Step 1: Generate expert dataset
    if not skip_sim:
        logging.info("Step 1 - Simulating track with pure pursuit to generate expert dataset")
        lap_time, expert_dataset, progress = simulate_track_pursuit(track)
        logging.info(f"Lap time: {lap_time:.2f} seconds")
        with open(EXPERT_DATASET_PATH, "wb") as f:
            pickle.dump(expert_dataset, f)
        logging.info(f"Saved expert dataset to {EXPERT_DATASET_PATH}")
    else:
        logging.info("Skipping simulation step")
        if not os.path.exists(EXPERT_DATASET_PATH):
            raise FileNotFoundError("Expert dataset not found but skip_sim=True.")
        with open(EXPERT_DATASET_PATH, "rb") as f:
            expert_dataset = pickle.load(f)

    # Step 2: Train Behaviour Cloning model
    if not skip_bc:
        logging.info("Step 2 - Training Behaviour Cloning model")
        if not expert_dataset or not isinstance(expert_dataset[0], (list, tuple)):
            raise ValueError("Expert dataset is empty or malformed")
        bc_model = train_bc(expert_dataset, epochs=BC_EPOCHS)
        torch.save(bc_model.state_dict(), BC_WEIGHTS_PATH)
        logging.info(f"Saved BC model weights to {BC_WEIGHTS_PATH}")
    else:
        logging.info("Skipping BC training step")
        if not os.path.exists(BC_WEIGHTS_PATH):
            raise FileNotFoundError("BC weights not found but skip_bc=True.")

    # Step 3: Initialise TD3 with BC warm start
    logging.info("Step 3 - Training TD3 with BC warm start")
    env = RacingEnv(track_name=TRACK_NAME, dt=0.1, discount_factor=DISCOUNT_FACTOR)
    model_bc = TD3("MlpPolicy", env, learning_rate=3e-4, verbose=1, batch_size=BATCH_SIZE)

    bc_actor = BCActor(input_dim=expert_dataset[0][0].shape[0],
                       output_dim=expert_dataset[0][1].shape[0])
    bc_actor.load_state_dict(torch.load(BC_WEIGHTS_PATH))
    with torch.no_grad():
        for bc_param, td3_param in zip(bc_actor.parameters(), model_bc.policy.actor.parameters()):
            td3_param.copy_(bc_param)
    logging.info("Loaded BC weights into TD3 actor.")

    # Train TD3
    model_bc.learn(total_timesteps=TD3_TIMESTEPS, progress_bar=True)
    model_bc.save(TD3_MODEL_PATH)
    logging.info(f"Saved TD3 model to {TD3_MODEL_PATH}")

    # Step 4: Visualise best lap
    logging.info("Step 4 - Visualising best lap")
    try:
        with open(BEST_LAP_LOG, "rb") as f:
            data = pickle.load(f)
            best_lap_time = data.get("best_lap_time")
            positions = data.get("positions")
            speeds = data.get("speeds")

            if best_lap_time and positions is not None and speeds is not None:
                logging.info(f"Best lap time: {best_lap_time:.2f} seconds")
                plot_track_and_trajectory(track, positions, speeds, best_lap_time)
            else:
                logging.warning("Incomplete data in best lap log; skipping visualisation.")
    except FileNotFoundError:
        logging.warning(f"{BEST_LAP_LOG} not found; skipping visualisation.")
    except Exception as e:
        logging.error(f"Failed to visualise best lap: {e}")

# === Direct Call Here ===
main(skip_sim=True, skip_bc=True)  # 👈 Adjust these flags as needed
