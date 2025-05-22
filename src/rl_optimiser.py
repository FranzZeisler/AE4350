import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from racing_env import RacingEnv
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

training_tracks = ["Austin", "Budapest", "Catalunya"]
testing_tracks = ["Spa", "Spielberg", "Suzuka", "YasMarina", "Zandvoort"]

# 👇 Predefined discount factors to try
discount_factors = [0.97, 0.975, 0.98, 0.985, 0.99, 0.995]

best_model_path = "best_model.zip"
best_lap_time = float("inf")
best_params = {}
lap_times_record = {}

def make_training_env(discount_factor):
    def make_env(track_name):
        return lambda: RacingEnv(
            track_name=track_name,
            dt=0.1,
            discount_factor=discount_factor,
        )
    return DummyVecEnv([make_env(t) for t in training_tracks])

if __name__ == "__main__":
    for discount in discount_factors:
        print(f"\n🔍 Evaluating discount_factor = {discount}")
        set_random_seed(42)
        env = make_training_env(discount)
        model = TD3("MlpPolicy", env, learning_rate=3e-4, verbose=1)
        model.learn(total_timesteps=1000000, progress_bar=True)

        lap_times = []

        for track in training_tracks:
            eval_env = RacingEnv(track_name=track, dt=0.1, discount_factor=discount)
            obs = eval_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, info = eval_env.step(action)
            if info.get("termination") == "lap_complete":
                lap_times.append(info["lap_time"])
            else:
                print(f"⚠️  Lap not completed on track '{track}' (termination: {info.get('termination')})")
                lap_times.append(999.0)

        avg_lap = np.mean(lap_times) if lap_times else 999.0
        lap_times_record[discount] = avg_lap
        print(f"📉 Avg Training Lap Time: {avg_lap:.2f}")

        if avg_lap < best_lap_time:
            best_lap_time = avg_lap
            best_params = {"discount_factor": discount}
            model.save(best_model_path)
            print("✅ New best model saved!")

    print("\n🏁 Best avg training lap time: {:.2f} seconds".format(best_lap_time))
    print("🏆 Best reward parameters:")
    for k, v in best_params.items():
        print(f" - {k}: {v}")

    # === Testing on unseen tracks ===
    model = TD3.load(best_model_path)

    for track_name in testing_tracks:
        print(f"\n🎥 Testing best model on unseen track: {track_name}")
        test_env = RacingEnv(
            track_name=track_name,
            dt=0.1,
            discount_factor=best_params["discount_factor"],
        )

        obs = test_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = test_env.step(action)

        test_env.render()
        print(f"🧪 Test Info: {info}")
