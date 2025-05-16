from matplotlib import pyplot as plt
import numpy as np
from car import Car
from pursuit_controller import pure_pursuit_control

def simulate_track_pursuit(track):
    car = Car(x=track["x_c"][0], y=track["y_c"][0], heading=track["heading"][0])
    path_points = np.stack((track["x_c"], track["y_c"]), axis=1)
    dt = car.dt
    max_time = 500.0  # seconds
    min_lap_time = 10.0  # ignore lap completion check before this time (to avoid early stop)
    lap_radius = 5.0  # meters to consider lap completed

    positions = []
    time_elapsed = 0.0

    while time_elapsed < max_time:
        steer = pure_pursuit_control(car.pos, car.heading, path_points)
         # === Constant throttle for now ===
        throttle = 1.0
        car.update(steer, throttle)
        positions.append(car.pos.copy())
        time_elapsed += dt

        # Check lap completion only after min_lap_time to avoid false positives
        if time_elapsed > min_lap_time:
            dist_to_start = np.linalg.norm(car.pos - path_points[0])
            if dist_to_start < lap_radius:
                print(f"Lap completed in {time_elapsed:.2f} seconds.")
                break

    positions = np.array(positions)

    # Plot results
    plt.figure(figsize=(10, 8))
    plt.plot(track["x_l"], track["y_l"], 'blue', label="Track boundaries")
    plt.plot(track["x_r"], track["y_r"], 'blue')
    plt.plot(positions[:, 0], positions[:, 1], 'r-', label="Car trajectory")
    plt.axis("equal")
    plt.legend()
    plt.title("Car following centerline with pure pursuit controller")
    plt.show()
