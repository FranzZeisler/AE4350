from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from stable_baselines3 import TD3
from stable_baselines3.common.logger import configure, CSVOutputFormat
from stable_baselines3.common.noise import NormalActionNoise
from behavior_cloning import BCActor
from racing_env import RacingEnv
from track import load_track
import torch
import pickle
import pandas as pd
import numpy as np
import os
import warnings

# === Choose the hyperparameter to tune ===
HYPERPARAM_TO_TUNE = "learning_rate" # "learning_rate" | "buffer_size" | "learning_starts" | "batch_size" | "tau" | "gamma" | "action_noise" | "seed"

# === Fixed Paths & Constants ===
TRACK_NAME = "Spielberg"
BC_WEIGHTS_PATH = f"bc_actor_{TRACK_NAME}.pth"
EXPERT_DATASET_PATH = f"{TRACK_NAME}_expert_dataset.pkl"
LOG_DIR = "./logs_td3_phase3"
FIGURES_DIR = "./td3_phase3_curves"

SCALING_FACTOR = 73.0
ALPHA = 0.62
FITNESS_FUNCTION = 3

TOTAL_TIMESTEPS = 200_000
SEED = 42

# === Hyperparameter Search Space Definitions ===
HYPERPARAM_SPACE = {
    "learning_rate": Real(1e-5, 1e-3, name="learning_rate", prior="log-uniform"),
    "buffer_size": Integer(50000, 1000000, name="buffer_size"),
    "learning_starts": Integer(100, 5000, name="learning_starts"),
    "batch_size": Integer(64, 512, name="batch_size"),
    "tau": Real(0.001, 0.02, name="tau", prior="uniform"),
    "gamma": Real(0.90, 0.999, name="gamma", prior="uniform"),
    "action_noise": Real(0.0, 0.2, name="action_noise", prior="uniform"),
    "seed": Integer(0, 100, name="seed"),
}

space = [HYPERPARAM_SPACE[HYPERPARAM_TO_TUNE]]

@use_named_args(space)
def objective(**params):
    value = params[HYPERPARAM_TO_TUNE]
    run_name = f"{TRACK_NAME}_{HYPERPARAM_TO_TUNE.upper()}{value}".replace(".", "")
    LOG_FILE = os.path.join(LOG_DIR, f"{run_name}.csv")
    print(f"\n▶ Running: {run_name}")

    with open(EXPERT_DATASET_PATH, "rb") as f:
        expert_dataset = pickle.load(f)
    input_dim = expert_dataset[0][0].shape[0]
    output_dim = expert_dataset[0][1].shape[0]

    env = RacingEnv(track_name=TRACK_NAME, scale=SCALING_FACTOR, alpha=ALPHA, fitness_function=FITNESS_FUNCTION)

    # Default TD3 parameters
    model_kwargs = {
        "learning_rate": params.get("learning_rate", 0.001),
        "buffer_size": params.get("buffer_size", 1000000),
        "learning_starts": params.get("learning_starts", 100),
        "batch_size": params.get("batch_size", 256),
        "tau": params.get("tau", 0.005),
        "gamma": params.get("gamma", 0.99),
        "action_noise": None,
        "seed": params.get("seed", SEED),
    }

    # Modify parameters based on the hyperparameter being tuned
    if HYPERPARAM_TO_TUNE == "action_noise":
        model_kwargs["action_noise"] = NormalActionNoise(mean=np.zeros(env.action_space.shape[0]), sigma=value * np.ones(env.action_space.shape[0]))
    else:
        model_kwargs[HYPERPARAM_TO_TUNE] = value
    
    # Initialise TD3 model with the hyperparameters
    model = TD3("MlpPolicy", env, verbose=0, **model_kwargs)

    os.makedirs(LOG_DIR, exist_ok=True)
    logger = configure(LOG_DIR)
    model.set_logger(logger)
    logger.output_formats = [CSVOutputFormat(LOG_FILE)]

    bc_actor = BCActor(input_dim, output_dim)
    bc_actor.load_state_dict(torch.load(BC_WEIGHTS_PATH))
    with torch.no_grad():
        for bc_param, td3_param in zip(bc_actor.parameters(), model.policy.actor.parameters()):
            td3_param.copy_(bc_param)

    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

    # Get best lap time
    best_lap_time = getattr(env, "best_lap_time", None)
    if best_lap_time is None or best_lap_time <= 0.0:
        print("⚠️ Warning: Invalid lap time detected. Assigning large penalty.")
        best_lap_time = 9999.0  # Penalty for failed runs
    else:
        print(f"🏁 Best lap time: {best_lap_time:.2f} seconds")

    df = pd.read_csv(LOG_FILE)
    df = df.dropna(subset=["rollout/ep_rew_mean", "time/episodes"])
    df["rollout/ep_rew_mean"] = pd.to_numeric(df["rollout/ep_rew_mean"])
    df["time/episodes"] = pd.to_numeric(df["time/episodes"])

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(df["time/episodes"], df["rollout/ep_rew_mean"], label="Episode Reward Mean")
    plt.xlabel("Episodes")
    plt.ylabel("Mean Reward")
    plt.title(f"TD3 Reward Curve - Tuning {HYPERPARAM_TO_TUNE}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(FIGURES_DIR, f"{run_name}.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"📈 Saved reward curve to {plot_path}")
    return best_lap_time

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    results = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=5,
        n_initial_points=3,
        random_state=SEED,
        verbose=True
    )

    print("\n🏁 Best Hyperparameter Found:")
    print(f"{HYPERPARAM_TO_TUNE}: {results.x[0]}")
