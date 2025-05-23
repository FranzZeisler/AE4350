import warnings
import numpy as np
from stable_baselines3 import TD3
from racing_env import RacingEnv
from simulate_pursuit import simulate_track_pursuit
from track import load_track

warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

if __name__ == "__main__":
    # List of tracks to simulate
    
    # Load the track
    track = load_track("Spielberg")
    
    # Run the Pursuit controller for the current track
    time, expert, progress = simulate_track_pursuit(track)
    
    # Print array shape
    print("Time elapsed:", time)
        
