import gym
from race_track_env import RaceTrackEnv  # Assuming your environment is in race_track_env.py
import numpy as np

# Initialize the environment (you can specify the track name and other parameters)
env = RaceTrackEnv(track_name="Austin", max_steps=500, gamma=0.95, dt=0.01)

# Reset the environment to start a new episode
state = env.reset()
done = False

# Run a loop for a certain number of steps (or until the episode is done)
step_count = 0
while not done:
    step_count += 1

    # Generate random actions (steering and throttle)
    # Steering: random between -1 and 1
    # Throttle: random between -1 and 1
    action = np.random.uniform(-1, 1, size=2)  # Action is a 2D vector (steering, throttle)
    
    # Take a step in the environment
    state, reward, done, info = env.step(action)
    
    # Print the current step, reward, and if the lap is finished
    print(f"Step {step_count}, Reward: {reward:.2f}, Lap Time: {env.lap_time if env.lap_time else 'N/A'}")
    
    # Render the environment (this will plot the track and the trajectory)
    if step_count % 50 == 0:  # You can change this to update the plot less frequently
        env.render()

# After finishing the episode
if env.lap_time:
    print(f"Final lap time: {env.lap_time:.2f} seconds")
else:
    print("Lap was not completed.")
    
env.close()
