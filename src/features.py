import numpy as np

def compute_lidar(car, track, angles=np.deg2rad([-90, -45, -15, 0, 15, 45, 90])):
    pos = car.pos
    heading = car.heading
    x_l, y_l = track["x_l"], track["y_l"]
    x_r, y_r = track["x_r"], track["y_r"]

    # Stack all boundary points
    walls = np.stack([np.concatenate([x_l, x_r]), np.concatenate([y_l, y_r])], axis=1)

    lidar_readings = []
    for angle in angles:
        ray_dir = np.array([np.cos(heading + angle), np.sin(heading + angle)])
        distances = np.linalg.norm(walls - pos, axis=1)
        projections = np.dot(walls - pos, ray_dir)
        mask = (projections > 0)
        filtered_dist = distances[mask]
        lidar_readings.append(np.min(filtered_dist) if len(filtered_dist) > 0 else 100.0)
    return np.array(lidar_readings)

def heading_errors(car, track_points):
    pos = car.pos
    heading = car.heading
    closest_idx = np.argmin(np.linalg.norm(track_points - pos, axis=1))
    next_idx = min(closest_idx + 3, len(track_points)-1)
    future_idx = min(closest_idx + 15, len(track_points)-1)

    def heading_to(idx):
        dx = track_points[idx][0] - pos[0]
        dy = track_points[idx][1] - pos[1]
        return (np.arctan2(dy, dx) - heading + np.pi) % (2 * np.pi) - np.pi

    return np.array([heading_to(next_idx), heading_to(future_idx)])

def curvature_features(track_points, current_idx, num=5):
    curvs = []
    for i in range(num):
        idx1 = min(current_idx + i, len(track_points) - 2)
        idx2 = idx1 + 1
        dx = track_points[idx2][0] - track_points[idx1][0]
        dy = track_points[idx2][1] - track_points[idx1][1]
        heading_diff = np.arctan2(dy, dx)
        curvs.append(heading_diff)
    return np.array(curvs)

def velocity_heading(car, track_heading):
    # Use the Car's existing velocity_heading() method
    return (car.velocity_heading() - track_heading + np.pi) % (2 * np.pi) - np.pi

def extract_features(car, track, path_points):
    lidar = compute_lidar(car, track)
    head_errs = heading_errors(car, path_points)

    closest_idx = np.argmin(np.linalg.norm(path_points - car.pos, axis=1))
    curvs = curvature_features(path_points, closest_idx)

    speed = car.speed
    lateral = car.last_lat_accel       # use attribute, no method
    centripetal = car.last_centripetal # use attribute, no method

    vel_angle = velocity_heading(car, track["heading"][closest_idx])

    return np.concatenate([lidar, head_errs, curvs, [speed, lateral, centripetal, vel_angle]])
