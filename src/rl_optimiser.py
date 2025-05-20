import numpy as np
from stable_baselines3 import PPO
from racing_env import RacingEnv
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Integer, Real

from stable_baselines3.common.utils import set_random_seed

# Define training/testing tracks
training_tracks = ["Austin"]
testing_tracks = ["Austin"]

# Define the search space for reward parameters
space = [
    Integer(-1000, -100, name="crash_penalty"),
    Integer(0, 500, name="lap_complete_reward"),
    Integer(0, 100, name="progress_reward_scale"),
    Integer(0, 10, name="acceleration_reward"),
    Integer(-20, 0, name="steering_penalty"),
    Real(0, 0.5, name="speed_reward"),
]

best_model_path = "best_model.zip"
best_lap_time = [float("inf")]
best_params = {}

@use_named_args(space)
def objective(**params):
    lap_times = []

    for track_name in training_tracks:
        env = RacingEnv(
            track_name=track_name,
            dt=0.1,
            crash_penalty=params["crash_penalty"],
            lap_complete_reward=params["lap_complete_reward"],
            progress_reward_scale=params["progress_reward_scale"],
            acceleration_reward=params["acceleration_reward"],
            steering_penalty=params["steering_penalty"],
            speed_reward=params["speed_reward"]
        )

        set_random_seed(42)
        model = PPO("MlpPolicy", env, verbose=0, seed=42)
        model.learn(total_timesteps=20000)

        for test_track in testing_tracks:
            test_env = RacingEnv(
                track_name=test_track,
                dt=0.1,
                crash_penalty=params["crash_penalty"],
                lap_complete_reward=params["lap_complete_reward"],
                progress_reward_scale=params["progress_reward_scale"],
                acceleration_reward=params["acceleration_reward"],
                steering_penalty=params["steering_penalty"],
                speed_reward=params["speed_reward"]
            )
            obs = test_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, info = test_env.step(action)

            if info.get("termination") == "lap_complete":
                lap_time = info["lap_time"]
            else:
                lap_time = 999.0
            lap_times.append(lap_time)

    avg_time = np.mean(lap_times) if lap_times else 999.0
    print(f"Average lap time: {avg_time:.2f} seconds")

    # Save best model and params
    if avg_time < best_lap_time[0]:
        best_lap_time[0] = avg_time
        model.save(best_model_path)
        best_params.clear()
        best_params.update(params)
        print("New best model saved.")

    return avg_time


if __name__ == "__main__":
    result = gp_minimize(
        objective,
        space,
        n_calls=10,
        n_initial_points=3,
        acq_func="EI",
        random_state=42,
        verbose=True,
    )

    print("\n✅ Best lap time: {:.2f} seconds".format(result.fun))
    print("🏁 Best reward parameters:")
    for name, val in zip([dim.name for dim in space], result.x):
        print(f" - {name}: {val}")

    # Load the saved best model and render it
    from stable_baselines3 import PPO
    model = PPO.load(best_model_path)

    for track_name in testing_tracks:
        print(f"\n🎥 Rendering best model on track: {track_name}")
        env = RacingEnv(
            track_name=track_name,
            dt=0.1,
            crash_penalty=best_params["crash_penalty"],
            lap_complete_reward=best_params["lap_complete_reward"],
            progress_reward_scale=best_params["progress_reward_scale"],
            acceleration_reward=best_params["acceleration_reward"],
            steering_penalty=best_params["steering_penalty"],
            speed_reward=best_params["speed_reward"]
        )

        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = env.step(action)

        env.render(race_line=True)  # this will call your custom plot function
        print(info)
