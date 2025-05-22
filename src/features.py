import numpy as np

def compute_rangefinders(car, track, num_rangefinders=10):
    """
    Simulate rangefinders by computing the distance from the car to the track boundary.
    :param car: The car object with its current position.
    :param track: The track object with boundary data.
    :param num_rangefinders: Number of rangefinders (measurements from the car’s front)
    :return: A list of distances from the car to the track boundary.
    """
    rangefinders = []
    for i in range(num_rangefinders):
        # Calculate distance to track boundary (simplified example)
        distance = np.sqrt((car.pos[0] - track.x_c[i])**2 + (car.pos[1] - track.y_c[i])**2)
        rangefinders.append(distance)
    return rangefinders

def compute_wall_contact_flag(car, track):
    """
    Determine if the car is in contact with the wall (off-track).
    :param car: The car object.
    :param track: The track object to check if the car is off-track.
    :return: 1 if the car is off-track, 0 otherwise.
    """
    if track.is_off_track(car.pos[0], car.pos[1]):
        return 1
    return 0

def compute_future_curvature(track, future_steps=5):
    """
    Compute the curvature of the track's centerline at future points.
    :param track: The track object containing the centerline.
    :param future_steps: Number of future points to sample for curvature.
    :return: A list of curvature values at future points.
    """
    curvatures = []
    for i in range(future_steps):
        curvature = 1 / (track.x_c[i] ** 2 + track.y_c[i] ** 2)  # Simplified curvature calculation
        curvatures.append(curvature)
    return curvatures

def extract_features(car, track):
    """
    Extract all features for the SAC policy network.
    :param car: The car object containing its state.
    :param track: The track object used to compute errors.
    :param prev_steering: The previous steering command.
    :param num_rangefinders: Number of rangefinders to simulate.
    :param future_steps: Number of steps ahead to consider for curvature.
    :return: A vector of features for input to the policy network.
    """
    # Compute individual features
    v_t = car.velocity  # Linear velocity
    a_t = car.linear_acceleration  # Linear acceleration (already calculated in the Car class)

    # Ensure track_progress is an integer index
    progress_index = int(np.clip(car.track_progress * len(track.heading), 0, len(track.heading) - 1))

    heading_error = car.heading - track.heading[progress_index]
    
    # Rangefinders: Simulate the distance from the car to the track boundaries
    rangefinders = compute_rangefinders(car, track)
    
    # Wall contact flag: Check if the car is off-track
    wt = compute_wall_contact_flag(car, track)
    
    # Future curvature: Compute the track's curvature ahead of the car
    c_t = compute_future_curvature(track)
    
    # Previous steering input
    delta_t_minus_1 = car.past_steering_angle  # Previous steering input
    
    # Track progress (normalized between 0 and 1)
    track_progress = car.track_progress  # Progress along the track
    
    # Combine all features into a single feature vector
    features = np.concatenate([v_t, [a_t], [heading_error], rangefinders, [delta_t_minus_1], [wt], c_t, [track_progress]])

    return features
