from matplotlib import pyplot as plt
import numpy as np
from shapely import LinearRing
from shapely.geometry import Point, Polygon
from car import Car
from pursuit_controller import pure_pursuit_control
from track import build_track_polygon
from visualisation import plot_track_and_trajectory

# Default values after running the pursuit_optimiser.py script for 250 iterations, 
# n_initial_points=50, and 14 training tracks
# It gave an average lap time of 77.64s on the training tracks and managed to complete all laps.
# It also completed all laps on the test tracks.

def simulate_track_pursuit(
    track,
    base_lookahead=6.8586,
    lookahead_gain=0.1362,
    alpha=0.4005,
    throttle_threshold_1=15.0,
    throttle_threshold_2=10.0,
    throttle_1=1.0,
    throttle_2=0.5171,
    throttle_3=0.6,
    plot_speed=False
):
    car = Car(x=track["x_c"][0], y=track["y_c"][0], heading=track["heading"][0])
    path_points = np.stack((track["x_c"], track["y_c"]), axis=1)
    track_polygon = build_track_polygon(track)

    dt = car.dt
    max_time = 500.0
    min_lap_time = 10.0
    lap_radius = 5.0

    positions = []
    speeds = []
    time_elapsed = 0.0
    crash_point = None

    car.max_steer_rate = np.deg2rad(50)

    while time_elapsed < max_time:
        lookahead_distance = base_lookahead + lookahead_gain * car.speed
        steer, lookahead_idx = pure_pursuit_control(car.pos, car.heading, path_points, lookahead_distance)

        vector_to_target = path_points[lookahead_idx] - car.pos
        angle_to_target = np.arctan2(vector_to_target[1], vector_to_target[0])
        heading_error = abs((angle_to_target - car.heading + np.pi) % (2 * np.pi) - np.pi)

        if heading_error < np.deg2rad(throttle_threshold_1):
            throttle = throttle_1
        elif heading_error < np.deg2rad(throttle_threshold_2):
            throttle = throttle_2
        else:
            throttle = throttle_3

        steer = alpha * steer + (1 - alpha) * car.steering_angle

        car.update(steer, throttle)
        positions.append(car.pos.copy())
        speeds.append(car.speed)
        time_elapsed += dt

        # Check track boundaries (include boundary as inside)
        car_point = Point(car.pos[0], car.pos[1])
        if not track_polygon.contains(car_point):
            #print("Car point is outside the track boundaries!")
            crash_point = car.pos.copy()
            time_elapsed = 999.0
            break

        # Check for lap completion
        if time_elapsed > min_lap_time:
            dist_to_start = np.linalg.norm(car.pos - path_points[0])
            if dist_to_start < lap_radius:
                print("Lap completed! in {:.2f}s".format(time_elapsed))
                break

    if plot_speed:
        plot_track_and_trajectory(track, positions, speeds=speeds, crash_point=crash_point)

    return time_elapsed
