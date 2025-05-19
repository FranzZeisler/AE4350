import numpy as np

def extract_features(car, track, path_points):
    """
    Simple feature extraction for the RacingEnv.

    Features:
    - Car position (x, y)
    - Car heading (angle)
    - Car speed

    Parameters:
    - car: Car object
    - track: track dictionary (unused here, but included for future extension)
    - path_points: np.array of track centerline points (unused here)

    Returns:
    - feature vector (numpy array)
    """
    x, y = car.pos
    heading = car.heading
    speed = car.speed

    features = np.array([x, y, heading, speed], dtype=np.float32)
    return features
