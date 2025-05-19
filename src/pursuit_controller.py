import numpy as np

def pure_pursuit_control(car_pos, car_heading, path_points, lookahead_distance=5.0):
    dists = np.linalg.norm(path_points - car_pos, axis=1)
    closest_idx = np.argmin(dists)

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
    vector_to_target = target_point - car_pos
    angle_to_target = np.arctan2(vector_to_target[1], vector_to_target[0])
    steering_angle = angle_to_target - car_heading
    steering_angle = (steering_angle + np.pi) % (2 * np.pi) - np.pi

    heading_error = abs(steering_angle)

    return steering_angle, heading_error, lookahead_idx

def compute_throttle(heading_error, throttle_threshold_1, throttle_threshold_2, throttle_1, throttle_2, throttle_3):
    if heading_error < np.deg2rad(throttle_threshold_1):
        return throttle_1
    elif heading_error < np.deg2rad(throttle_threshold_2):
        return throttle_2
    else:
        return throttle_3

def smooth_steering(new_steer, current_steer, alpha):
    return alpha * new_steer + (1 - alpha) * current_steer
