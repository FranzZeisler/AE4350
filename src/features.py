# features.py
import numpy as np

def extract_features(car, track, path):
    """
    Make a feature vector from the car's state and the track. 
    This feature vector is used for training the RL agent.
    The feature vector contains minimal information about the car's state
    It incudes the car's speed, acceleration, steering angle, and the distance to the track centerline.
    """
    # 1) Car state
    speed = car.speed
    steer = car.steering_angle
    lat_accel = car.last_lat_accel
    centripetal = car.last_centripetal
    vel_heading = car.last_vel_heading

    # Distance to track centerline
    track_pos = np.array([car.pos[0] - track["x_c"][0], car.pos[1] - track["y_c"][0]])
    track_pos = np.linalg.norm(track_pos)  # Euclidean distance to centerline
    track_pos = np.clip(track_pos, -10.0, 10.0)  # Limit the distance to a reasonable range
    
    # 3) Feature vector
    feature_vector = np.concatenate((track_pos, [speed, steer, lat_accel, centripetal, vel_heading]))
    
    return feature_vector
    
