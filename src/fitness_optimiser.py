from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from stable_baselines3 import TD3
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise
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
TOTAL_TIMESTEPS = 200_000
SEED = 42

# === Search Space ===
space = [
    Real(0.975, 0.999, name="discount_factor"),
    Real(1.0, 100.0, name="scaling_factor"),
]

@use_named_args(space)
def objective(discount_factor, scaling_factor):
    run_name = f"{TRACK_NAME}_DF{discount_factor:.4f}_SF{scaling_factor:.2f}".replace(".", "")
    TD3_MODEL_PATH = f"td3_{run_name}.zip"
    LOG_DIR = f"./logs_td3_{run_name}"
    LOG_FILE = f"{LOG_DIR}/progress.csv"

    print(f"\n▶ Running: discount_factor={discount_factor:.4f}, scale={scaling_factor:.2f}")

    # Load expert dataset
    with open(EXPERT_DATASET_PATH, "rb") as f:
        expert_dataset = pickle.load(f)
    input_dim = expert_dataset[0][0].shape[0]
    output_dim = expert_dataset[0][1].shape[0]

    # Initialise environment
    env = RacingEnv(track_name=TRACK_NAME, discount_factor=discount_factor, scale=scaling_factor)

    # Init model
    model = TD3("MlpPolicy", env, seed=SEED, verbose=0)

    # Configure logger
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = configure(LOG_DIR, ["csv"])
    model.set_logger(logger)
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
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    model.save(TD3_MODEL_PATH)

    # Evaluate
    try:
        df = pd.read_csv(LOG_FILE)
        df = df.dropna(subset=["rollout/ep_rew_mean"])
        reward = df["rollout/ep_rew_mean"].iloc[-1]
        print(f"🎯 Final reward: {reward:.2f}")
        return -reward  # We minimise
    except Exception as e:
        print(f"⚠️ Failed to read log: {e}")
        return 1e6

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