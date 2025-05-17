import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from car import Car
from neural_controller import NeuralController
from track import load_track

def build_track_polygon(track):
    """
    Build a polygon representing the track boundaries.
    :param track: The track object containing left and right boundaries.
    :return: A Shapely Polygon object representing the track.
    """
    left_boundary = np.stack((track["x_l"], track["y_l"]), axis=1)
    right_boundary = np.stack((track["x_r"], track["y_r"]), axis=1)[::-1]  # reversed to close polygon
    polygon_points = np.vstack((left_boundary, right_boundary))
    return Polygon(polygon_points)

def compute_centerline_progress(track):
    """
    Compute the centerline and cumulative lengths of the track segments.
    :param track: The track object containing centerline points.
    :return: A tuple containing the centerline points and cumulative lengths.
    """
    centerline = np.stack((track["x_c"], track["y_c"]), axis=1)
    diffs = np.diff(centerline, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)
    return centerline, cumulative_lengths

def simulate_track_neural(genome, track, render=True, return_progress=False):
    """
    Simulate the car on the track using a neural network controller.
    :param genome: The genome (weights and biases) of the neural network.
    :param track: The track object containing the track data.
    :param render: Whether to render the simulation.
    :param return_progress: Whether to return the progress made.
    :return: A tuple containing the fitness score and time elapsed.
    """
    car = Car(x=track["x_c"][0], y=track["y_c"][0], heading=track["heading"][0])
    dt = car.dt
    max_time = 800.0

    track_polygon = build_track_polygon(track)
    centerline, cumulative_lengths = compute_centerline_progress(track)
    track_length = cumulative_lengths[-1]

    nn = NeuralController(genome)

    fitness = 0.0
    off_track_penalty = 0.1
    lateral_penalty_coeff = 10.0
    progress_reward_coeff = 100.0
    lap_completion_bonus = 2.0

    last_progress = 0.0
    positions = []
    time_elapsed = 0.0

    while time_elapsed < max_time:
        obs = car.get_feature_vector(track, centerline)
        steer, throttle = nn.forward(obs)

        car.update(steer, throttle)
        positions.append(car.pos.copy())
        time_elapsed += dt

        car_point = Point(car.pos[0], car.pos[1])
        if not track_polygon.contains(car_point):
            fitness *= off_track_penalty
            break

        distances = np.linalg.norm(centerline - car.pos, axis=1)
        closest_idx = np.argmin(distances)

        current_progress = cumulative_lengths[closest_idx]
        progress_delta = current_progress - last_progress
        if progress_delta > 0:
            fitness += progress_reward_coeff * progress_delta
            last_progress = current_progress
                
        lateral_error = np.linalg.norm(car.pos - centerline[closest_idx])
        fitness -= lateral_penalty_coeff * lateral_error

        # Early finish if lap complete
        if current_progress >= 0.99 * track_length:
            fitness *= lap_completion_bonus
            print(f"Lap completed at time {time_elapsed:.2f}s with fitness {fitness:.2f}")
            break
    
    if render:
        # Visualize the trajectory
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

    # Return fitness and time elapsed
    # If return_progress is True, also return the progress made as a fraction of the track length
    if return_progress:
        return fitness, time_elapsed, last_progress / track_length
    else:
        return fitness, time_elapsed
