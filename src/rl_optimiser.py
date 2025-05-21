import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Integer, Real
from racing_env import RacingEnv
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

# TRAIN on these
training_tracks = ["Austin", "Budapest", "Catalunya"]
testing_tracks = ["Spa", "Spielberg", "Suzuka", "YasMarina", "Zandvoort"]

# Define search space for reward parameters
space = [
    Integer(0, 500, name="acceleration_reward"),
]

best_model_path = "best_model.zip"
best_lap_time = [float("inf")]
best_params = {}

# Helper: create vec environment for all training tracks
def make_training_env(params):
    def make_env(track_name):
        return lambda: RacingEnv(
            track_name=track_name,
            dt=0.1,
            acceleration_reward=params["acceleration_reward"],
            steering_penalty=params["steering_penalty"],
            speed_reward=params["speed_reward"]
        )
    return DummyVecEnv([make_env(t) for t in training_tracks])

@use_named_args(space)
def objective(**params):
    print("\n🔍 Trying reward parameters:")
    for key, value in params.items():
        print(f" - {key}: {value}")
    
    env = make_training_env(params)
    set_random_seed(42)
    model = PPO("MlpPolicy", env, verbose=0, seed=42)
    model.learn(total_timesteps=20000)

    lap_times = []

    # Evaluate on training tracks (not testing tracks!)
    for track in training_tracks:
        eval_env = RacingEnv(
            track_name=track,
            dt=0.1,
            acceleration_reward=params["acceleration_reward"],
            steering_penalty=params["steering_penalty"],
            speed_reward=params["speed_reward"]
        )

        obs = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = eval_env.step(action)

        lap_time = info.get("lap_time", 999.0) if info.get("termination") == "lap_complete" else 999.0
        lap_times.append(lap_time)

    avg_lap = np.mean(lap_times) if lap_times else 999.0
    print(f"\n📉 Avg Training Lap Time: {avg_lap:.2f}")

    if avg_lap < best_lap_time[0]:
        best_lap_time[0] = avg_lap
        model.save(best_model_path)
        best_params.clear()
        best_params.update(params)
        print("✅ New best model saved!")

    return avg_lap

if __name__ == "__main__":
    result = gp_minimize(
        objective,
        space,
        n_calls=20,
        n_initial_points=5,
        acq_func="EI",
        random_state=42,
        verbose=True,
    )

    print("\n🏁 Best avg training lap time: {:.2f} seconds".format(result.fun))
    print("🏆 Best reward parameters:")
    for name, val in zip([dim.name for dim in space], result.x):
        print(f" - {name}: {val}")

    # Load best model and test on generalization tracks
    model = PPO.load(best_model_path)

    for track_name in testing_tracks:
        print(f"\n🎥 Testing best model on unseen track: {track_name}")
        test_env = RacingEnv(
            track_name=track_name,
            dt=0.1,
            acceleration_reward=best_params["acceleration_reward"],
            steering_penalty=best_params["steering_penalty"],
            speed_reward=best_params["speed_reward"]
        )

        obs = test_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = test_env.step(action)

        test_env.render()
        print(f"🧪 Test Info: {info}")
