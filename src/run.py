import warnings
import numpy as np
from stable_baselines3 import TD3
from racing_env import RacingEnv
from simulate_pursuit import simulate_track_pursuit
from track import load_track

warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

if __name__ == "__main__":
    # List of tracks to simulate
    track_names = [ "Austin", "Budapest", "Catalunya", "Hockenheim", "Melbourne", 
    "MexicoCity", "Montreal", "Monza", "Sakhir", "SaoPaulo", "Sepang", "Shanghai", 
    "Silverstone", "Sochi", "Spa", "Spielberg", "Suzuka", "YasMarina", "Zandvoort"]
    
    # Initialize the overall max steer_normalised variable
    max_steer_normalised_overall = -float('inf')  # Start with a very low value

    # Iterate over all tracks and find the maximum steer_normalised
    for track_name in track_names:
        print(f"\nSimulating track: {track_name}")
        
        # Load the track
        track = load_track(track_name)
        
        # Run the Pursuit controller for the current track
        time, expert = simulate_track_pursuit(track)
        
        # Print the time elapsed for this track
        print(f"Time elapsed for {track_name}: {time:.2f} seconds")
        
        # Find the maximum steer_normalised for this track
        max_steer_normalised_for_track = -float('inf')  # Start with a very low number for each track
        
        # Iterate through expert dataset to find the maximum steer_normalised
        for obs, action in expert:
            steer_normalised = action[0]  # Assuming steer_normalised is the first element in the action
            if steer_normalised > max_steer_normalised_for_track:
                max_steer_normalised_for_track = steer_normalised
        
        # Print the max steer_normalised for this track
        print(f"Maximum steer_normalised for {track_name}: {max_steer_normalised_for_track:.5f}")
        
        # Update the overall maximum steer_normalised if this track's max is higher
        if max_steer_normalised_for_track > max_steer_normalised_overall:
            max_steer_normalised_overall = max_steer_normalised_for_track
    
    # Print the overall maximum steer_normalised across all tracks
    print(f"\nOverall maximum steer_normalised across all tracks: {max_steer_normalised_overall:.5f}")
