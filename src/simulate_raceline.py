import numpy as np
from visualisation import plot_track_and_trajectory
from car import Car  # Only used for car specs (e.g., max_lateral_accel)
from track import build_track_polygon
from scipy.interpolate import splprep, splev


def compute_curvature(path):
    """
    Compute the curvature of a path using finite differences.
    :param path: Nx2 array of (x, y) points
    :return: Nx1 array of curvature values
    """
    dx = np.gradient(path[:, 0])
    dy = np.gradient(path[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    curvature[np.isnan(curvature)] = 0.0
    return curvature

def resample_path(path, spacing=0.5):
    """
    Resample a path to have approximately uniform spacing between points.
    :param path: Nx2 array of (x, y) points
    :param spacing: desired distance between points
    :return: resampled Nx2 path
    """
    x, y = path[:, 0], path[:, 1]
    tck, u = splprep([x, y], s=0, per=False)
    length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    num_points = int(length / spacing)
    u_fine = np.linspace(0, 1, num_points)
    x_fine, y_fine = splev(u_fine, tck)
    return np.vstack((x_fine, y_fine)).T

def simulate_raceline(track, plot_speed=False):
    """
    Simulate the raceline on a track and return the total time taken.
    :param track: Track data containing x_c, y_c, and raceline.
    :param plot_speed: Whether to plot the track and trajectory.
    :return: Total time taken to complete the raceline.
    """
    
    # Check if the track has a raceline
    if track["raceline"] is None:
        raise ValueError("Track has no raceline!")

    # Resample the raceline to have uniform spacing
    raceline = track["raceline"]
    raceline = resample_path(raceline, spacing=0.5)

    # Use Car class to access vehicle limits
    car = Car(x=track["x_c"][0], y=track["y_c"][0], heading=track["heading"][0])
    max_lateral_accel = car.max_lateral_accel
    max_speed = car.max_speed

    # Precompute speed at each raceline point
    curvature = compute_curvature(raceline)
    with np.errstate(divide='ignore'):
        max_speeds = np.sqrt(np.where(curvature > 1e-6, max_lateral_accel / curvature, max_speed))
    max_speeds = np.clip(max_speeds - 2.0, 0.0, max_speed)

    # Resample the raceline to have uniform spacing
    positions = []
    speeds = []
    total_time = 0.0

    # Simulate the raceline
    # Loop through each segment of the raceline
    for i in range(len(raceline) - 1):
        p0 = raceline[i]
        p1 = raceline[i + 1]

        # Calculate the distance and heading for this segment.
        segment = p1 - p0
        distance = np.linalg.norm(segment)
        if distance < 1e-6:
            continue
        heading = np.arctan2(segment[1], segment[0])
        speed = max_speeds[i]
        dt = distance / speed

        # "Move" the car to this position and speed
        pos = p1
        positions.append(pos.copy())
        speeds.append(speed)
        total_time += dt

    # If plot_speed is True, plot the track and trajectory
    if plot_speed:
        plot_track_and_trajectory(track, positions, speeds=speeds)

    return total_time
