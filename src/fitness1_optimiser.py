import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from stable_baselines3 import TD3
from stable_baselines3.common.logger import configure
from stable_baselines3.common.logger import CSVOutputFormat
from behavior_cloning import BCActor
from racing_env import RacingEnv
from track import load_track
import torch
import pickle
import pandas as pd
import numpy as np
import warnings
import os

# === Fixed Paths & Constants ===
TRACK_NAME = "Spielberg"
BC_WEIGHTS_PATH = f"bc_actor_{TRACK_NAME}.pth"
EXPERT_DATASET_PATH = f"{TRACK_NAME}_expert_dataset.pkl"
FIGURES_DIR = "./td3_learning_curves"
LOG_DIR = "./logs_td3"

FITNESS_FUNCTION = 1
TOTAL_TIMESTEPS = 200_000
SEED = 42

# === Search Space ===
space = [
    Real(0.975, 0.999, name="discount_factor"),
    Integer(1.0, 100.0, name="scaling_factor"),
]

@use_named_args(space)
def objective(discount_factor, scaling_factor):
    run_name = f"{TRACK_NAME}_DF{discount_factor:.4f}_SF{scaling_factor:.2f}".replace(".", "")
    LOG_FILE = os.path.join(LOG_DIR, f"{run_name}.csv")

    print(f"\n▶ Running: discount_factor={discount_factor:.4f}, scale={scaling_factor:.2f}")

    # Load expert dataset
    with open(EXPERT_DATASET_PATH, "rb") as f:
        expert_dataset = pickle.load(f)
    input_dim = expert_dataset[0][0].shape[0]
    output_dim = expert_dataset[0][1].shape[0]

    # Initialise environment
    env = RacingEnv(track_name=TRACK_NAME, discount_factor=discount_factor, scale=scaling_factor, fitness_function=FITNESS_FUNCTION)

    # Init model
    model = TD3("MlpPolicy", env, seed=SEED, verbose=0)

    # Configure logger (single folder, multiple CSVs)
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = configure(LOG_DIR)  # Use default setup (older API only supports the folder)
    model.set_logger(logger)
    logger.output_formats = [CSVOutputFormat(LOG_FILE)]

    logger.record("config/track_name", TRACK_NAME)
    logger.record("config/discount_factor", discount_factor)
    logger.record("config/scaling_factor", scaling_factor)
    logger.dump(step=0)

    # Load BC weights into TD3 actor
    bc_actor = BCActor(input_dim=input_dim, output_dim=output_dim)
    bc_actor.load_state_dict(torch.load(BC_WEIGHTS_PATH))
    with torch.no_grad():
        for bc_param, td3_param in zip(bc_actor.parameters(), model.policy.actor.parameters()):
            td3_param.copy_(bc_param)
    print("✅ Loaded BC weights into TD3 actor.")

    # Train
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

    # Print best lap time
    best_lap_time = getattr(env, "best_lap_time", 0.0)
    print(f"🏁 Best lap time: {best_lap_time:.2f} seconds")

    # Plot and save learning curve
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

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_name = f"{TRACK_NAME}_DF{discount_factor:.4f}_SF{scaling_factor:.2f}_Lap{best_lap_time:.2f}".replace(".", "")
    fig_name += ".png"
    fig_path = os.path.join(FIGURES_DIR, fig_name)
    plt.savefig(fig_path)
    plt.close()

    print(f"📈 Saved learning curve to {fig_path}\n")
    print("------------------------------------------------")

    return -df["rollout/ep_rew_mean"].iloc[-1]

# === Run Bayesian Optimisation ===
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    results = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=10,
        n_initial_points=5,
        random_state=42,
        verbose=True,
    )

    print("\n🏁 Best Parameters Found:")
    print(f"Discount Factor: {results.x[0]:.4f}")
    print(f"Scaling Factor:  {results.x[1]:.2f}")

