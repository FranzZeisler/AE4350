
import stable_baselines3 as sb3
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from race_track_env import RaceTrackEnv
from track import Track

# Initialize track and environment
track_name = "Spielberg"  # Replace with actual track loading logic
track = Track(track_name)
env = RaceTrackEnv(track=track, track_width=10.0)

# Define SAC model parameters
policy_kwargs = dict(net_arch=[256, 256], activation_fn=sb3.common.torch_layers.ReLU)
model = SAC("MlpPolicy", env, 
            gamma=0.99, 
            learning_rate=3e-4, 
            buffer_size=1000000, 
            batch_size=256, 
            tau=0.005, 
            train_freq=1, gradient_steps=1, 
            ent_coef='auto', target_entropy='auto',
            policy_kwargs=policy_kwargs, 
            verbose=1, device='auto')

# Train the model
model.learn(total_timesteps=2_000_000, log_interval=1000)

# Save the trained model
model.save("sac_racing_model")
