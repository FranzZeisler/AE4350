import numpy as np

def angle_diff(a, b):
    """Compute minimal signed difference between two angles (radians)."""
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi

def extract_features(car, track):
    """
    Extract minimal feature vector for RL agent.

    Features:
    - normalized speed
    - distance to centerline
    - distance to left boundary
    - distance to right boundary
    - heading error (car heading vs track heading)
    - curvature at nearest centerline point
    - normalized steering angle
    
    :param car: Car object
    :param track: Track dict with centerline and boundaries
    :return: np.ndarray of shape (7,)
    """
    pos = car.pos
    x_c, y_c = track["x_c"], track["y_c"]
    x_l, y_l = track["x_l"], track["y_l"]
    x_r, y_r = track["x_r"], track["y_r"]
    headings = track["heading"]  # radians
    
    # 1) Find nearest centerline point index
    distances = np.linalg.norm(np.column_stack((x_c, y_c)) - pos, axis=1)
    idx = np.argmin(distances)
    
    # 2) Distance to centerline (signed approx)
    # Vector from centerline to car
    vec_c_to_car = pos - np.array([x_c[idx], y_c[idx]])
    # Track heading at idx points along the track direction
    track_heading = headings[idx]
    # Perp vector to track heading (pointing left)
    perp = np.array([-np.sin(track_heading), np.cos(track_heading)])
    dist_centerline = np.dot(vec_c_to_car, perp)  # signed distance: + left side, - right side
    
    # 3) Distances to left and right boundaries at idx
    dist_left = np.linalg.norm(pos - np.array([x_l[idx], y_l[idx]]))
    dist_right = np.linalg.norm(pos - np.array([x_r[idx], y_r[idx]]))
    
    # 4) Heading error (car heading - track heading)
    hdg_err = angle_diff(car.heading, track_heading)
    
    # 5) Curvature approx from track heading differences (centerline)
    # Use forward difference for simplicity; last point same curvature as before
    headings_roll = np.roll(headings, -1)
    d_heading = angle_diff(headings_roll[idx], headings[idx])
    # Approximate curvature = delta heading / delta s
    # Approximate segment length ds = distance between idx and idx+1
    if idx < len(x_c) - 1:
        ds = np.linalg.norm(np.array([x_c[idx+1], y_c[idx+1]]) - np.array([x_c[idx], y_c[idx]]))
        curvature = d_heading / ds if ds > 0 else 0.0
    else:
        curvature = 0.0
    
    # 6) Normalize speed and steering
    speed_norm = car.speed / car.max_speed
    steer_norm = car.steering_angle / car.max_steering_angle
    
    # Compose feature vector
    features = np.array([
        speed_norm,
        dist_centerline,
        dist_left,
        dist_right,
        hdg_err,
        curvature,
        steer_norm,
    ], dtype=np.float32)
    
    return features
