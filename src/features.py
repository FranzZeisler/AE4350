import numpy as np
from shapely.geometry import Point

def compute_lidar(car, track, angles=np.deg2rad([-90, -45, -15, 0, 15, 45, 90]), max_range=100.0):
    """
    Compute Lidar-like readings based on the car's position and heading.
    :param car: The car object with position and heading.
    :param track: The track object containing the left and right boundaries.
    :param angles: Angles in radians for the Lidar rays relative to car heading.
    :param max_range: Max lidar range in meters.
    :return: Lidar readings as an array of distances (clipped to max_range).
    """
    pos = car.pos
    heading = car.heading
    x_l, y_l = track["x_l"], track["y_l"]
    x_r, y_r = track["x_r"], track["y_r"]

    # Stack all boundary points (walls)
    walls = np.stack([np.concatenate([x_l, x_r]), np.concatenate([y_l, y_r])], axis=1)

    lidar_readings = []
    for angle in angles:
        ray_dir = np.array([np.cos(heading + angle), np.sin(heading + angle)])

        # Vector from car to all walls
        rel_vecs = walls - pos

        # Projection of wall vectors onto ray direction
        projections = np.dot(rel_vecs, ray_dir)
        mask = (projections > 0)

        if not np.any(mask):
            lidar_readings.append(max_range)
            continue

        # Distances only for walls in front of the ray
        filtered_vecs = rel_vecs[mask]
        distances = np.linalg.norm(filtered_vecs, axis=1)

        lidar_readings.append(min(np.min(distances), max_range))

    return np.array(lidar_readings)

def find_lookahead_index(path_points, start_idx, distance):
    """
    Find the index in path_points approximately distance meters ahead starting from start_idx.
    """
    total_dist = 0.0
    idx = start_idx
    while total_dist < distance and idx < len(path_points) - 1:
        seg = path_points[idx + 1] - path_points[idx]
        seg_len = np.linalg.norm(seg)
        total_dist += seg_len
        idx += 1
    return idx

def compute_curvature_at(path_points, idx, window=3):
    """
    Approximate curvature at a given index by calculating heading difference over arc length.
    """
    i0 = max(idx - window, 0)
    i1 = min(idx + window, len(path_points) - 1)

    p0, p1 = path_points[i0], path_points[i1]
    dx, dy = p1 - p0
    heading = np.arctan2(dy, dx)

    # For curvature, estimate heading change over arc length
    # Here simplified: curvature ≈ heading / arc length
    arc_len = np.linalg.norm(p1 - p0)
    if arc_len == 0:
        return 0.0
    return heading / arc_len

def velocity_heading(car, track_heading):
    # relative velocity heading error: velocity heading - track heading, normalized
    return (car.velocity_heading() - track_heading + np.pi) % (2 * np.pi) - np.pi


def extract_features(car, track, path_points, lookahead_distances=[5, 10, 20, 50, 100]):
    """
    Extract a combined feature vector from the car and track state.
    Features include:
     - Ego state: speed, lateral accel, centripetal accel, velocity heading, steering angle
     - Target info: relative positions and heading errors at multiple lookahead distances
     - Curvature at multiple lookahead distances
     - Normalized lidar readings
    
    :param car: Car object
    :param track: Track dictionary with "x_l", "y_l", "x_r", "y_r", "heading"
    :param path_points: Nx2 numpy array of track centerline points
    :param lookahead_distances: list of distances (m) to extract target info and curvature
    :return: 1D numpy array feature vector
    """
    lidar = compute_lidar(car, track)
    closest_idx = np.argmin(np.linalg.norm(path_points - car.pos, axis=1))

    # Ego state
    speed = car.speed
    lateral = car.last_lat_accel
    centripetal = car.last_centripetal
    vel_angle = velocity_heading(car, track["heading"][closest_idx])
    steering_angle = car.steering_angle

    ego_state = np.array([speed, lateral, centripetal, vel_angle, steering_angle])

    # Target info: relative positions (car frame) and heading errors at multiple distances
    target_positions = []
    heading_errors = []
    for d in lookahead_distances:
        idx = find_lookahead_index(path_points, closest_idx, d)
        target_pos = path_points[idx]

        # Relative position in world frame
        rel_pos = target_pos - car.pos

        # Transform to car frame (rotate by -heading)
        c, s = np.cos(-car.heading), np.sin(-car.heading)
        x_car = c * rel_pos[0] - s * rel_pos[1]
        y_car = s * rel_pos[0] + c * rel_pos[1]
        target_positions.extend([x_car, y_car])

        # Heading error to target point
        dx, dy = target_pos - car.pos
        angle_to_target = np.arctan2(dy, dx)
        head_err = (angle_to_target - car.heading + np.pi) % (2 * np.pi) - np.pi
        heading_errors.append(head_err)

    target_info = np.array(target_positions + heading_errors)

    # Curvature features at lookahead distances
    curvatures = []
    for d in lookahead_distances:
        idx = find_lookahead_index(path_points, closest_idx, d)
        curv = compute_curvature_at(path_points, idx)
        curvatures.append(curv)
    curvatures = np.array(curvatures)

    # Normalize lidar (assuming max range 100m)
    lidar_norm = np.clip(lidar / 100.0, 0, 1)

    features = np.concatenate([ego_state, target_info, curvatures, lidar_norm])
    return features
