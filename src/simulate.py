import numpy as np
import matplotlib.pyplot as plt
from car import Car
from pursuit_controller import pure_pursuit_control
from track import load_track

def simulate_track(track):
    car = Car(x=track["x_c"][0], y=track["y_c"][0], heading=track["heading"][0])
    path_points = np.stack((track["x_c"], track["y_c"]), axis=1)
    dt = car.dt
    max_time = 500.0
    min_lap_time = 10.0
    lap_radius = 5.0

    positions = []
    time_elapsed = 0.0

    while time_elapsed < max_time:
        # Pure pursuit provides steering; we manually set throttle
        steer = pure_pursuit_control(car.pos, car.heading, path_points)
        throttle = 1.0  # full throttle for now; replace with logic if needed

        car.update(steer, throttle)
        positions.append(car.pos.copy())
        time_elapsed += dt

        # Check lap completion
        dist_to_start = np.linalg.norm(car.pos - path_points[0])
        if time_elapsed > min_lap_time and dist_to_start < lap_radius:
            print(f"Lap completed in {time_elapsed:.2f} seconds.")
            break

    # Plot
    positions = np.array(positions)
    plt.figure(figsize=(10, 8))
    plt.plot(track["x_l"], track["y_l"], 'blue', label="Track boundaries")
    plt.plot(track["x_r"], track["y_r"], 'blue')
    plt.plot(positions[:, 0], positions[:, 1], 'r-', label="Car trajectory")
    plt.axis("equal")
    plt.legend()
    plt.title("Car with Pure Pursuit Controller and Enhanced Dynamics")
    plt.show()

if __name__ == "__main__":
    track = load_track("Austin")  # or any other available track
    simulate_track(track)
