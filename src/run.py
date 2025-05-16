from simulate import simulate_track
from track import load_track

if __name__ == "__main__":
    track = load_track("Silverstone")
    simulate_track(track)
