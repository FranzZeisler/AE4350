import numpy as np

# ---------------------
# F1 Car Physical Constants (Tuned for smooth centerline tracking)
# ---------------------

wheelbase = 3.6                   # [m] Typical modern F1 car wheelbase
max_accel = 15.0                  # [m/s²] Max acceleration (traction limited)
max_decel = 30.0                  # [m/s²] Max braking force
max_lateral_g = 4.0 * 9.81        # [m/s²] Reduced lateral G-limit for safer cornering (~4G)
top_speed = 100.0                 # [m/s] Approx 360 km/h max speed
dt = 0.005                       # [s] Smaller timestep for smoother simulation

max_steer_deg_per_step = 2.0     # Reduced max heading change allowed per timestep (degrees)

# ---------------------
# Reactive Car Agent (Centerline tracking with anti-oscillation)
# ---------------------

def car_agent(v_car, pos_car, dis2wall_r, dis2wall_l, dis2centerline, center_curve, hdg_nxt, genome):
    """
    Simulates one timestep of an F1 car using reactive control focused on centerline tracking
    with measures to reduce oscillations.

    Inputs:
    - v_car: current velocity vector [vx, vy]
    - pos_car: current position vector [x, y]
    - dis2wall_r, dis2wall_l: distances to right/left track edges
    - dis2centerline: distance to track centerline
    - center_curve: array of centerline points [[x1,y1], [x2,y2], ...]
    - hdg_nxt: array of headings (radians) between centerline points
    - genome: list of 6 reactive tuning parameters:
        [0] steer_gain: how aggressively it corrects heading error
        [1] centerline_tol: tolerance to being off-center (not used in this version)
        [2] heading_lookahead: future turn severity threshold (used for speed control)
        [3] future_idx_short: lookahead index (short-term)
        [4] future_idx_long: lookahead index (long-term)
        [5] accel_gain: throttle aggression level

    Returns:
    - v_car: new velocity vector [vx, vy]
    - nearest_centerpoint: closest [x,y] on centerline
    - index_closest: index of that point
    - spd: updated speed scalar
    - hdg_car_deg: current heading in degrees
    """

    steer_gain, centerline_tol, heading_tol, future_short, future_long, accel_gain = genome

    # Find closest centerline point
    distances = np.linalg.norm(center_curve - pos_car, axis=1)
    index_closest = np.argmin(distances)
    nearest_center = center_curve[index_closest]

    n = len(center_curve)
    i_short = (index_closest + int(future_short)) % n

    # Current speed and heading
    spd = np.linalg.norm(v_car) + 1e-9  # avoid div by zero
    hdg_car_rad = np.arctan2(v_car[1], v_car[0])
    hdg_car_deg = np.degrees(hdg_car_rad) % 360

    # Target heading based on lookahead point on centerline
    hdg_next = np.degrees(hdg_nxt[i_short]) % 360

    # Heading difference (signed [-180,180])
    heading_error = ((hdg_next - hdg_car_deg + 180) % 360) - 180

    # Steering correction based on lateral offset with deadzone to avoid jitter
    deadzone = 0.05  # meters

    if dis2centerline < deadzone:
        steer_correction = 0.0
    else:
        # Determine side of offset using cross product sign
        vec_to_center = nearest_center - pos_car
        car_direction = v_car / spd
        cross = car_direction[0]*vec_to_center[1] - car_direction[1]*vec_to_center[0]
        side_sign = np.sign(cross)

        steer_correction = steer_gain * dis2centerline * side_sign

    # Clamp steering correction to max degrees per timestep
    steer_correction = np.clip(steer_correction, -max_steer_deg_per_step, max_steer_deg_per_step)

    new_heading_deg = (hdg_car_deg + steer_correction) % 360

    # Speed control based on heading error
    heading_change = abs(heading_error)
    if heading_change > heading_tol and spd > 10:
        spd = max(spd - max_decel * dt, 0)
    elif spd < top_speed:
        spd += max_accel * dt * accel_gain

    spd = min(spd, top_speed)

    # Update velocity vector
    new_heading_rad = np.radians(new_heading_deg)
    v_car = np.array([spd * np.cos(new_heading_rad), spd * np.sin(new_heading_rad)])

    return v_car, nearest_center, index_closest, spd, new_heading_deg
