import numpy as np


def pure_pursuit_control(car_pos, car_heading, path_points, lookahead_distance=5.0):
    """
    Find target point on path at lookahead distance and calculate steering.

    car_pos: np.array([x, y])
    car_heading: float, radians
    path_points: Nx2 np.array of track centerline points
    lookahead_distance: float in meters
    """

    # Find closest point index on path
    dists = np.linalg.norm(path_points - car_pos, axis=1)
    closest_idx = np.argmin(dists)

    # Find target point at lookahead distance ahead on path
    path_len = path_points.shape[0]
    lookahead_idx = closest_idx
    total_dist = 0.0
    while total_dist < lookahead_distance:
        next_idx = (lookahead_idx + 1) % path_len
        segment = path_points[next_idx] - path_points[lookahead_idx]
        seg_len = np.linalg.norm(segment)
        total_dist += seg_len
        lookahead_idx = next_idx

    target_point = path_points[lookahead_idx]

    # Calculate steering angle to target point
    vector_to_target = target_point - car_pos
    angle_to_target = np.arctan2(vector_to_target[1], vector_to_target[0])
    steering_angle = angle_to_target - car_heading

    # Wrap steering angle to [-pi, pi]
    steering_angle = (steering_angle + np.pi) % (2 * np.pi) - np.pi

    return steering_angle
