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
    :return: A tuple containing the fitness score, time elapsed.
    """
    car = Car(x=track["x_c"][0], y=track["y_c"][0], heading=track["heading"][0])
    dt = car.dt
    max_time = 200.0

    track_polygon = build_track_polygon(track)
    centerline, cumulative_lengths = compute_centerline_progress(track)
    track_length = cumulative_lengths[-1]

    nn = NeuralController(genome)

    fitness = 0.0

    max_allowed_error = 3  # meters, roughly half the track width minus some margin
    reward_coeff = 10.0      # fitness reward scale for staying close to centerline
    off_track_penalty = 20.0 # penalty for going off the allowed error range
    lap_completion_bonus = 1000.0

    last_progress = 0.0
    positions = []
    time_elapsed = 0.0

    while time_elapsed < max_time:
        #print(f"Time: {time_elapsed:.2f}s, Position: {car.pos}, Speed: {car.speed:.2f}, Heading: {car.heading:.2f}, Fitness: {fitness:.2f}")
        obs = car.get_feature_vector(track, centerline)
        steer, throttle = nn.forward(obs)

        car.update(steer, throttle)
        positions.append(car.pos.copy())
        time_elapsed += dt

        car_point = Point(car.pos[0], car.pos[1])
        if not track_polygon.contains(car_point):
            # Penalize and stop simulation if car goes outside track polygon
            fitness -= off_track_penalty
            print("Car went off track polygon. Ending simulation.")
            break

        # Find the closest point on the centerline and calculate lateral error
        distances = np.linalg.norm(centerline - car.pos, axis=1)
        closest_idx = np.argmin(distances)
        current_progress = cumulative_lengths[closest_idx]
        lateral_error = distances[closest_idx]

        if lateral_error <= max_allowed_error:
            # Reward for staying close to centerline (linear from max reward at 0m error to 0 at max_allowed_error)
            fitness += reward_coeff * (max_allowed_error - lateral_error) / max_allowed_error
        else:
            # Penalize if car goes too far from centerline and end simulation early
            fitness -= off_track_penalty
            #print(f"Car lateral error {lateral_error:.2f}m exceeded max allowed {max_allowed_error}m. Ending simulation.")
            break

        # Optionally reward progress (comment out if you want only centerline reward)
        progress_delta = current_progress - last_progress
        if progress_delta > 0:
            # Small bonus for forward progress
            fitness += progress_delta * 1.0
            last_progress = current_progress

        # Early finish if lap complete
        if current_progress >= 0.99 * track_length:
            print(f"Lap completed at time {time_elapsed:.2f}s with fitness {fitness:.2f}")
            fitness += lap_completion_bonus
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
        plt.title("Neural Network Controller Trajectory (Pursuit-Style Fitness)")
        plt.show()

    if return_progress:
        return fitness, time_elapsed, current_progress / track_length
    else:
        return fitness, time_elapsed
