from matplotlib import pyplot as plt
import numpy as np
from shapely import LinearRing
from shapely.geometry import Point, Polygon
from car import Car
from pursuit_controller import pure_pursuit_control

def build_track_polygon(track):
    left_boundary = np.column_stack((track["x_l"], track["y_l"]))
    right_boundary = np.column_stack((track["x_r"], track["y_r"]))[::-1]  # reversed

    # Use LinearRing to ensure closed, simple rings
    left_ring = LinearRing(left_boundary)
    right_ring = LinearRing(right_boundary)

    # Construct polygon from left + right rings
    polygon_points = np.vstack((left_ring.coords, right_ring.coords))
    poly = Polygon(polygon_points)

    # Fix polygon if invalid
    if not poly.is_valid:
        poly = poly.buffer(0)
        #print("Warning: Track polygon was invalid and was fixed.")

    return poly

def simulate_track_pursuit(
    track,
    base_lookahead=3.0,
    lookahead_gain=0.4,
    alpha=0.5,
    throttle_threshold_1=5.0,
    throttle_threshold_2=20.0,
    throttle_1=1.0,
    throttle_2=0.7,
    throttle_3=0.3,
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
                break

    if plot_speed:
        positions = np.array(positions)
        speeds = np.array(speeds)

        plt.figure(figsize=(10, 8))
        plt.plot(track["x_l"], track["y_l"], 'r-', label="Left boundary")
        plt.plot(track["x_r"], track["y_r"], 'b-', label="Right boundary")

        # Plot trajectory colored by speed
        sc = plt.scatter(positions[:, 0], positions[:, 1], c=speeds, cmap='jet', s=5)
        cbar = plt.colorbar(sc)
        cbar.set_label("Speed (m/s)")

        # Mark crash location if it happened
        if crash_point is not None:
            plt.plot(crash_point[0], crash_point[1], 'rx', markersize=14, label="Crash ❌")

        plt.axis("equal")
        plt.legend()
        plt.title("Car following centerline with pure pursuit controller")
        plt.show()

    return time_elapsed
