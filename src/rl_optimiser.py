from sklearn.model_selection import ParameterGrid
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from racing_env import RacingEnv  # Replace with your actual racing environment
import warnings

# Suppress warnings from stable_baselines3
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

# Define the training and testing tracks
training_tracks = [
    "Austin", "Budapest", "Catalunya", "Hockenheim", "Melbourne", 
    "MexicoCity", "Montreal", "Monza", "Sakhir", "SaoPaulo",
    "Sepang", "Shanghai", "Silverstone", "Sochi"
]
testing_tracks = ["Spa", "Spielberg", "Suzuka", "YasMarina", "Zandvoort"]

# Define the hyperparameter space for learning rates
param_grid = {
    'learning_rate': [3e-4, 1e-4, 1e-3]  # Learning rates to experiment with
}

best_model_path = "best_model.zip"
best_lap_time = float("inf")
best_params = {}

# Placeholder for storing test results
test_lap_times = {track: [] for track in testing_tracks}

# Function to create the training environment
def make_training_env():
    def make_env(track_name):
        return lambda: RacingEnv(track_name=track_name)
    return DummyVecEnv([make_env(t) for t in training_tracks])

# Loop through each set of hyperparameters in the parameter grid
for params in ParameterGrid(param_grid):
    lr = params['learning_rate']
    print(f"\n🔍 Evaluating with learning rate: {lr}")
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Create the training environment
    env = make_training_env()
    
    # Initialize and train the TD3 model with the current learning rate
    model = TD3("MlpPolicy", env, learning_rate=lr, verbose=0)
    model.learn(total_timesteps=50000, progress_bar=True, log_interval=10)  # Fixed 2000 timesteps for training

    # Evaluate the model's performance on the training tracks
    lap_times = []

    for track in training_tracks:
        eval_env = RacingEnv(track_name=track)
        obs = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = eval_env.step(action)
        if info.get("termination") == "lap_complete":
            lap_times.append(info["lap_time"])
        else:
            print(f"⚠️ Lap not completed on track '{track}' (termination: {info.get('termination')})")
            lap_times.append(999.0)

    avg_lap = np.mean(lap_times) if lap_times else 999.0
    print(f"📉 Avg Training Lap Time: {avg_lap:.2f}")

    # Save the best model based on training performance
    if avg_lap < best_lap_time:
        best_lap_time = avg_lap
        best_params = params
        model.save(best_model_path)
        print("✅ New best model saved!")

# Report the best hyperparameters
print("\n🏁 Best avg training lap time: {:.2f} seconds".format(best_lap_time))
print("🏆 Best reward parameters:")
for k, v in best_params.items():
    print(f" - {k}: {v}")

# === Testing on unseen tracks ===
# Load the best model
model = TD3.load(best_model_path)

# Test on unseen tracks
for track_name in testing_tracks:
    print(f"\n🎥 Testing best model on unseen track: {track_name}")
    test_env = RacingEnv(track_name=track_name)

    obs = test_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = test_env.step(action)

    test_env.render()  # Optional, can be omitted if running in headless mode
    print(f"🧪 Test Info for track {track_name}: {info}")
    
    # Collect lap time info for test track
    if info.get("termination") == "lap_complete":
        test_lap_times[track_name].append(info["lap_time"])
    else:
        print(f"⚠️ Lap not completed on track '{track_name}' (termination: {info.get('termination')})")
        test_lap_times[track_name].append(999.0)

# Calculate and report average test lap times
print("\n🏁 Test Results:")
for track, lap_times in test_lap_times.items():
    avg_test_lap = np.mean(lap_times) if lap_times else 999.0
    print(f" - {track}: Avg Test Lap Time: {avg_test_lap:.2f} seconds")
