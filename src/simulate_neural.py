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

def compute_centerline_progress(track):
    centerline = np.stack((track["x_c"], track["y_c"]), axis=1)
    diffs = np.diff(centerline, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)
    return centerline, cumulative_lengths

def simulate_track_neural(genome, track, render=True, return_progress=False):
    car = Car(x=track["x_c"][0], y=track["y_c"][0], heading=track["heading"][0])
    dt = car.dt
    max_time = 1500.0

    track_polygon = build_track_polygon(track)
    centerline, cumulative_lengths = compute_centerline_progress(track)
    track_length = cumulative_lengths[-1]

    nn = NeuralController(genome)
    fitness = 0.0

    last_progress = 0.0
    positions = []
    time_elapsed = 0.0

    while time_elapsed < max_time:
        obs = extract_features(car, track, centerline)
        steer, throttle = nn.forward(obs)
        throttle = max(0.0, throttle)  # no reverse
        car.update(steer, throttle)
        positions.append(car.pos.copy())
        time_elapsed += dt

        # Off-track check - immediate penalty and stop
        car_point = Point(car.pos[0], car.pos[1])
        if not track_polygon.contains(car_point):
            fitness -= 10000  # heavy penalty
            break

        # Nearest centerline point
        distances = np.linalg.norm(centerline - car.pos, axis=1)
        closest_idx = np.argmin(distances)
        current_progress = cumulative_lengths[closest_idx]

        # Reward progress, ramp up reward the further along
        progress_delta = current_progress - last_progress
        if progress_delta > 0:
            progress_fraction = current_progress / track_length
            multiplier = 1 + 2 * progress_fraction  # ramps 1 → 3
            fitness += progress_delta * multiplier
            last_progress = current_progress

            # Bonus for being near centerline (within 0–2 meters gives full reward)
            lateral_error = np.linalg.norm(car.pos - centerline[closest_idx])
            centerline_bonus = max(0, 1 - (lateral_error / 2)) * progress_delta * 0.5  # 0.5 weight
            fitness += centerline_bonus

        # Optional: finish early if lap complete
        if current_progress >= 0.95 * track_length:
            fitness += 10000  # bonus for finishing lap
            break

    if render:
        positions = np.array(positions)
        plt.figure(figsize=(10, 8))
        plt.plot(track["x_l"], track["y_l"], 'blue', label="Track boundaries")
        plt.plot(track["x_r"], track["y_r"], 'blue')
        plt.plot(track["x_c"], track["y_c"], 'k--', label="Centerline", alpha=0.5)
        plt.plot(positions[:, 0], positions[:, 1], 'r-', label="Car trajectory")
        plt.axis("equal")
        plt.legend()
        plt.title("Neural Network Controller Trajectory")
        plt.show()

    progress_ratio = last_progress / track_length
    if return_progress:
        return fitness, time_elapsed, progress_ratio
    return fitness, time_elapsed
