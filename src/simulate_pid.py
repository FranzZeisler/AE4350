import numpy as np
import matplotlib.pyplot as plt
from car import Car
from pid_controller import pid_steering_control, PIDController
from track import load_track

def simulate_track_pid(track):
    car = Car(x=track["x_c"][0], y=track["y_c"][0], heading=track["heading"][0])
    path_points = np.stack((track["x_c"], track["y_c"]), axis=1)
    dt = car.dt
    max_time = 500.0
    min_lap_time = 10.0
    lap_radius = 5.0

    positions = []
    time_elapsed = 0.0

    # === Initialize the steering PID controller ===
    steering_pid = PIDController(kp=2.5, ki=0.0, kd=0.5)

    while time_elapsed < max_time:
        # === Compute heading error to lookahead point ===
        heading_error = pid_steering_control(car, path_points)

        # === Use PID to convert heading error into a steering command ===
        steer = steering_pid.control(heading_error, dt)

        # === Constant throttle for now ===
        throttle = 1.0

        # === Update car state ===
        car.update(steer, throttle)
        positions.append(car.pos.copy())
        time_elapsed += dt

        # === Lap completion check ===
        dist_to_start = np.linalg.norm(car.pos - path_points[0])
        if time_elapsed > min_lap_time and dist_to_start < lap_radius:
            print(f"Lap completed in {time_elapsed:.2f} seconds.")
            break

    # === Plot ===
    positions = np.array(positions)
    plt.figure(figsize=(10, 8))
    plt.plot(track["x_l"], track["y_l"], 'blue', label="Track boundaries")
    plt.plot(track["x_r"], track["y_r"], 'blue')
    plt.plot(positions[:, 0], positions[:, 1], 'r-', label="Car trajectory")
    plt.axis("equal")
    plt.legend()
    plt.title("Car with PID Controller and Enhanced Dynamics")
    plt.show()
