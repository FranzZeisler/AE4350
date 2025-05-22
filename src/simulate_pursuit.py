from matplotlib import pyplot as plt
import numpy as np
from shapely.geometry import Point
from car import Car
from pursuit_controller import pure_pursuit_control, compute_throttle, smooth_steering
from track import build_track_polygon
from visualisation import plot_track_and_trajectory

def simulate_track_pursuit(
    track,
    base_lookahead=10,
    lookahead_gain=0.05,
    alpha=0.25,
    throttle_threshold_1=20.0,
    throttle_1=1.0,
    throttle_2=0.8,
    plot_speed=False
):
    """
    Simulate the car on the track using pure pursuit control.
    Args:
        track (dict): Track data containing x_c, y_c, and heading.
        base_lookahead (float): Base lookahead distance for pure pursuit.
        lookahead_gain (float): Gain for lookahead distance based on speed.
        alpha (float): Smoothing factor for steering input.
        throttle_threshold_1 (float): First throttle threshold in degrees.
        throttle_1 (float): Throttle value for the first range.
        throttle_2 (float): Throttle value for the second range.
        plot_speed (bool): Whether to plot the track and trajectory.
    Returns:
        float: Time elapsed during the simulation.
        list: Expert dataset of (observation, action) tuples.
    """

    # Initialize car at the start of the centerline
    car = Car(x=track["x_c"][0], y=track["y_c"][0], heading=track["heading"][0])
    # Path points for pure pursuit
    path_points = np.stack((track["x_c"], track["y_c"]), axis=1)
    # Build track polygon for collision detection
    track_polygon = build_track_polygon(track)

    # Simulation parameters
    dt = car.dt
    max_time = 500.0
    min_lap_time = 10.0
    lap_radius = 5.0

    # Initialize variables for simulation & plotting
    positions = []
    speeds = []
    time_elapsed = 0.0
    crash_point = None

    # Expert dataset to store (obs, action)
    expert_dataset = []

    # Simulation loop
    # Loop until max_time is reached or car crashes
    while time_elapsed < max_time:
        # Calculate the lookahead distance based on speed
        lookahead_distance = base_lookahead + lookahead_gain * car.speed

        # Pure pursuit control gives steering and throttle
        steer, heading_error = pure_pursuit_control(
            car.pos, car.heading, path_points, lookahead_distance
        )

        # Calculate throttle based on heading error
        throttle = compute_throttle(
            heading_error,
            throttle_threshold_1,
            throttle_1,
            throttle_2,
        )

        # Smooth the steering input
        steer = smooth_steering(steer, car.steering_angle, alpha)
        # Limit steering to [-1, 1]
        steer_clipped = np.clip(steer, -car.max_steering_angle, car.max_steering_angle)
        steer_normalised = steer_clipped / car.max_steering_angle  
        #print(f"Pursuit Raw Steering: {steer:.5f}, Pursuit Raw Throttle: {throttle:.5f}")


        # Get observation vector from the car for RL
        obs = car.get_feature_vector(track)
        # Create action vector
        action = np.array([steer_normalised, throttle])
        # Append to expert dataset
        expert_dataset.append((obs, action))

        # Update car state
        car.update(steer, throttle)
        positions.append(car.pos.copy())
        speeds.append(car.speed)
        time_elapsed += dt

        # Crash check: is the car within track boundaries?
        car_point = Point(car.pos[0], car.pos[1])
        if not track_polygon.contains(car_point):
            crash_point = car.pos.copy()
            time_elapsed = 999.0  # Penalty for crash
            break

        # Check for lap completion
        if time_elapsed > min_lap_time:
            dist_to_start = np.linalg.norm(car.pos - path_points[0])
            if dist_to_start < lap_radius:
                #print("Lap completed! in {:.2f}s".format(time_elapsed))
                break

    # If plot_speed is True, plot the track and trajectory
    if plot_speed:
        plot_track_and_trajectory(track, positions, speeds=speeds, crash_point=crash_point)

    # Return the time elapsed and expert dataset
    return time_elapsed, expert_dataset
