import numpy as np

def pure_pursuit_control(car_pos, car_heading, path_points, lookahead_distance=5.0):
    dists = np.linalg.norm(path_points - car_pos, axis=1)
    closest_idx = np.argmin(dists)

    path_len = path_points.shape[0]
    lookahead_idx = closest_idx
    total_dist = 0.0
    max_steps = 1000  # safeguard

    steps = 0
    while total_dist < lookahead_distance and steps < max_steps:
        next_idx = (lookahead_idx + 1) % path_len
        segment = path_points[next_idx] - path_points[lookahead_idx]
        seg_len = np.linalg.norm(segment)
        total_dist += seg_len
        lookahead_idx = next_idx
        steps += 1

    target_point = path_points[lookahead_idx]
    vector_to_target = target_point - car_pos
    angle_to_target = np.arctan2(vector_to_target[1], vector_to_target[0])
    steering_angle = angle_to_target - car_heading
    steering_angle = (steering_angle + np.pi) % (2 * np.pi) - np.pi

    return steering_angle, lookahead_idx

def compute_local_curvature(path_points, idx, window=5):
    """
    Compute approximate curvature by measuring heading change over a window.
    """
    i0 = max(idx - window, 0)
    i1 = min(idx + window, len(path_points) - 1)
    p0, p1 = path_points[i0], path_points[i1]
    dx, dy = p1 - p0
    heading = np.arctan2(dy, dx)
    return heading
