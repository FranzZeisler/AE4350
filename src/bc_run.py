import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import logging
import warnings
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.logger import configure
from behavior_cloning import train_bc, BCActor
from racing_env import RacingEnv
from simulate_pursuit import simulate_track_pursuit
from track import load_track
from visualisation import plot_track_and_trajectory

# === Suppress Warnings ===
warnings.filterwarnings("ignore", category=UserWarning)

# === Logging Setup ===
logging.basicConfig(filename='log.txt', filemode='w', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# === Configurable Parameters ===
TRACK_NAME = "Spielberg"
DISCOUNT_FACTOR = 0.98
SCALING_FACTOR = 1.0
ALPHA = 0.5
BC_EPOCHS = 300

#=== File Paths ===
BC_WEIGHTS_PATH = f"bc_actor_{TRACK_NAME}.pth"
EXPERT_DATASET_PATH = f"{TRACK_NAME}_expert_dataset.pkl"
TD3_MODEL_PATH = f"td3_{TRACK_NAME}_with_bc.zip"
BEST_LAP_LOG = "log_best_lap_time.pkl"
LOG_DIR = "./logs_td3"
LOG_FILE = "logs_td3/progress.csv"

# === TD3 Hyperparameters ===
LEARNING_RATE = 0.001
BUFFER_SIZE = 1000000
LEARNING_STARTS = 100
BATCH_SIZE = 256
TAU = 0.005
GAMMA = 0.99
ACTION_NOISE_STDDEV = 0.0
SEED = 42

TD3_TIMESTEPS = 200

def main(skip_sim=False, skip_bc=False):
    logging.info(f"Loading track: {TRACK_NAME}")
    track = load_track(TRACK_NAME)

    # Step 1: Generate expert dataset
    if not skip_sim:
        logging.info("Step 1 - Simulating track with pure pursuit to generate expert dataset")
        lap_time, expert_dataset, progress = simulate_track_pursuit(track)
        logging.info(f"Step 1 - Lap time: {lap_time:.2f} seconds")
        with open(EXPERT_DATASET_PATH, "wb") as f:
            pickle.dump(expert_dataset, f)
        logging.info(f"Step 1 - Saved expert dataset to {EXPERT_DATASET_PATH}")
    else:
        logging.info("Step 1 - Skipping simulation step")
        if not os.path.exists(EXPERT_DATASET_PATH):
            raise FileNotFoundError("Expert dataset not found but skip_sim=True.")
        with open(EXPERT_DATASET_PATH, "rb") as f:
            expert_dataset = pickle.load(f)

    # Step 2: Train Behaviour Cloning model
    if not skip_bc:
        logging.info("Step 2 - Training Behaviour Cloning model")
        if not expert_dataset or not isinstance(expert_dataset[0], (list, tuple)):
            raise ValueError("Expert dataset is empty or malformed")
        bc_model = train_bc(expert_dataset, epochs=BC_EPOCHS, verbose=0)
        torch.save(bc_model.state_dict(), BC_WEIGHTS_PATH)
        logging.info(f"Step 2 - Saved BC model weights to {BC_WEIGHTS_PATH}")
    else:
        logging.info("Step 2 - Skipping BC training step")
        if not os.path.exists(BC_WEIGHTS_PATH):
            raise FileNotFoundError("BC weights not found but skip_bc=True.")
        
    # Step 3: Evaluate BC policy before TD3 training
    logging.info("Step 3 - Evaluating BC policy before TD3 training")
    env = RacingEnv(track_name=TRACK_NAME, dt=0.1, discount_factor=DISCOUNT_FACTOR, scale=SCALING_FACTOR, alpha=ALPHA)
    input_dim = expert_dataset[0][0].shape[0]
    output_dim = expert_dataset[0][1].shape[0]
    bc_actor = BCActor(input_dim, output_dim)
    bc_actor.load_state_dict(torch.load(BC_WEIGHTS_PATH))
    bc_actor.eval()
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0
    while not done:
        action = bc_actor(torch.FloatTensor(obs)).detach().numpy()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
    lap_time = info.get("lap_time", -1)
    logging.info(f"Step 3 - BC lap complete: {lap_time:.2f} seconds, total reward: {total_reward:.2f}, steps: {steps}")
    env.render()

    # Step 4: Initialise TD3 with BC warm start
    logging.info("Step 4 - Initialising TD3 with BC warm start")
    env = RacingEnv(track_name=TRACK_NAME, dt=0.1, discount_factor=DISCOUNT_FACTOR, scale=SCALING_FACTOR, alpha=ALPHA)

    # Add action noise for TD3
    action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape[0]), sigma=ACTION_NOISE_STDDEV * np.ones(env.action_space.shape[0]))

    model_bc = TD3(
        policy="MlpPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        tau=TAU,
        gamma=GAMMA,
        action_noise=action_noise,
        verbose=0,
        seed=SEED,
    )

    # Setup CSV logger
    new_logger = configure(LOG_DIR, ["csv"])
    model_bc.set_logger(new_logger)
    # Log TD3 and config parameters (excluding paths)
    new_logger.record("config/track_name", TRACK_NAME)
    new_logger.record("config/discount_factor", DISCOUNT_FACTOR)
    new_logger.record("config/scaling_factor", SCALING_FACTOR)
    new_logger.record("config/bc_epochs", BC_EPOCHS)
    new_logger.record("config/td3_timesteps", TD3_TIMESTEPS)
    new_logger.record("config/learning_rate", LEARNING_RATE)
    new_logger.record("config/buffer_size", BUFFER_SIZE)
    new_logger.record("config/learning_starts", LEARNING_STARTS)
    new_logger.record("config/batch_size", BATCH_SIZE)
    new_logger.record("config/tau", TAU)
    new_logger.record("config/gamma", GAMMA)
    new_logger.record("config/action_noise_stddev", ACTION_NOISE_STDDEV)
    new_logger.record("config/seed", SEED)

    # Flush to ensure it's written before training
    new_logger.dump(step=0)

    # Load BC weights into TD3 actor
    bc_actor = BCActor(input_dim=expert_dataset[0][0].shape[0], output_dim=expert_dataset[0][1].shape[0])
    bc_actor.load_state_dict(torch.load(BC_WEIGHTS_PATH))
    with torch.no_grad():
        for bc_param, td3_param in zip(bc_actor.parameters(), model_bc.policy.actor.parameters()):
            td3_param.copy_(bc_param)
    logging.info("Step 4 - Finished loading BC weights into TD3 actor.")

    # Step 5: Train TD3 with BC warm start
    logging.info("Step 5 - Training TD3 with BC warm start")
    model_bc.learn(total_timesteps=TD3_TIMESTEPS, progress_bar=True)
    model_bc.save(TD3_MODEL_PATH)
    logging.info(f"Step 5 - Completed training of TD3 model and saved model to {TD3_MODEL_PATH}")

    # Show plot of rewards vs timesteps
    df = pd.read_csv(LOG_FILE)
    df = df.dropna(subset=["rollout/ep_rew_mean", "time/episodes"])
    df["rollout/ep_rew_mean"] = pd.to_numeric(df["rollout/ep_rew_mean"])
    df["time/episodes"] = pd.to_numeric(df["time/episodes"])
    plt.figure(figsize=(10, 6))
    plt.plot(df["time/episodes"], df["rollout/ep_rew_mean"], label="Episode Reward Mean", marker='o')
    plt.xlabel("Episodes")
    plt.ylabel("Mean Reward")
    plt.title("TD3 Learning Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Step 6: Visualise best lap after training
    logging.info("Step 6 - Visualising best lap")
    try:
        with open(BEST_LAP_LOG, "rb") as f:
            data = pickle.load(f)
            best_lap_time = data.get("best_lap_time")
            positions = data.get("positions")
            speeds = data.get("speeds")
            if best_lap_time and positions is not None and speeds is not None:
                logging.info(f"Step 6 - Best lap time: {best_lap_time:.2f} seconds")
                plot_track_and_trajectory(track, positions, speeds)
            else:
                logging.info("Step 6 - Incomplete data in best lap log; skipping visualisation.")
    except FileNotFoundError:
        logging.info(f"Step 6 - {BEST_LAP_LOG} not found; skipping visualisation.")
    except Exception as e:
        logging.info(f"Step 6 - Failed to visualise best lap: {e}")

# === Direct Call Here ===
main(skip_sim=True, skip_bc=True)
