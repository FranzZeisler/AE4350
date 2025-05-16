import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from car import Car
from neural_controller import NeuralController
from features import extract_features
from track import load_track

def build_track_polygon(track):
    left_boundary = np.stack((track["x_l"], track["y_l"]), axis=1)
    right_boundary = np.stack((track["x_r"], track["y_r"]), axis=1)[::-1]  # reversed to close polygon
    polygon_points = np.vstack((left_boundary, right_boundary))
    return Polygon(polygon_points)

def compute_track_length(track):
    x_c, y_c = track["x_c"], track["y_c"]
    diffs = np.diff(np.stack((x_c, y_c), axis=1), axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return np.sum(distances)

def simulate_track_neural(genome, track, render=True):
    car = Car(x=track["x_c"][0], y=track["y_c"][0], heading=track["heading"][0])
    path_points = np.stack((track["x_c"], track["y_c"]), axis=1)
    dt = car.dt
    max_time = 500.0
    min_lap_time = 10.0
    lap_radius = 5.0

    track_polygon = build_track_polygon(track)
    track_length = compute_track_length(track)

    positions = []
    time_elapsed = 0.0
    nn = NeuralController(genome)

    fitness = 0.0
    off_track_penalty = 1000.0

    distance_traveled = 0.0
    last_pos = car.pos.copy()

    while time_elapsed < max_time:
        obs = extract_features(car, track, path_points)
        steer, throttle = nn.forward(obs)
        throttle = max(0.0, throttle)  # prevent backward motion
        car.update(steer, throttle)
        positions.append(car.pos.copy())
        time_elapsed += dt

        # Accumulate traveled distance
        distance_traveled += np.linalg.norm(car.pos - last_pos)
        last_pos = car.pos.copy()

        # Check if car inside track polygon
        car_point = Point(car.pos[0], car.pos[1])
        if not track_polygon.contains(car_point):
            fitness -= off_track_penalty
            print("Car went off track!")
            break

        # Reward for moving forward (speed * dt)
        fitness += car.speed * dt

        # Lap complete?
        dist_to_start = np.linalg.norm(car.pos - path_points[0])
        if (
            time_elapsed > min_lap_time
            and dist_to_start < lap_radius
            and distance_traveled >= 0.95 * track_length
        ):
            fitness += 10000  # Big bonus for lap completion
            print(f"Lap completed in {time_elapsed:.2f} seconds.")
            break

    if render:
        positions = np.array(positions)
        plt.figure(figsize=(10, 8))
        plt.plot(track["x_l"], track["y_l"], 'blue', label="Track boundaries")
        plt.plot(track["x_r"], track["y_r"], 'blue')
        plt.plot(positions[:, 0], positions[:, 1], 'r-', label="Car trajectory")
        plt.axis("equal")
        plt.legend()
        plt.title("Neural Network Controller Trajectory")
        plt.show()

    return fitness, time_elapsed
