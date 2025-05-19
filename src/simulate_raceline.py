import numpy as np
from visualisation import plot_track_and_trajectory
from car import Car  # Only used for car specs (e.g., max_lateral_accel)
from track import build_track_polygon
from scipy.interpolate import splprep, splev


def compute_curvature(path):
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
    if track["raceline"] is None:
        raise ValueError("Track has no raceline!")

    raceline = track["raceline"]
    raceline = resample_path(raceline, spacing=0.5)
    polygon = build_track_polygon(track)

    # Use Car class to access vehicle limits
    dummy_car = Car(0, 0, 0)
    max_lateral_accel = dummy_car.max_lateral_accel
    max_speed = dummy_car.max_speed

    # Precompute speed at each raceline point
    curvature = compute_curvature(raceline)
    with np.errstate(divide='ignore'):
        max_speeds = np.sqrt(np.where(curvature > 1e-6,
                                      max_lateral_accel / curvature,
                                      max_speed))
    max_speeds = np.clip(max_speeds - 2.0, 0.0, max_speed)

    positions = []
    speeds = []
    total_time = 0.0

    for i in range(len(raceline) - 1):
        p0 = raceline[i]
        p1 = raceline[i + 1]

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


    if plot_speed:
        plot_track_and_trajectory(track, positions, speeds=speeds)

    return total_time
