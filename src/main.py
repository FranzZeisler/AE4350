# Initialize environment
from stable_baselines3 import SAC
from race_track_env import RaceTrackEnv


env = RaceTrackEnv(track_name="Austin")  # Use the appropriate track name

# Define the SAC model
model = SAC("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10, progress_bar=True)    

# Save the trained model
model.save("sac_model")

# Evaluate the trained model
model = SAC.load("sac_model")

obs = env.reset()
for _ in range(5):  # Simulate for 1000 steps
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
env.render()  # Visualize the car's performance