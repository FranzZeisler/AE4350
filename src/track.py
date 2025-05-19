import pandas as pd
import numpy as np
import os
from shapely.geometry import LinearRing, Polygon


def load_track(track_name, base_path="data"):
    """
    Load and process a racetrack from the TUMFTM dataset.

    Parameters:
    - track_name: str
        Name of the CSV file (without .csv)
    - base_path: str
        Base path to the 'data' directory

    Returns:
    - dict containing:
        - x_c, y_c: centerline coordinates
        - x_l, y_l: left boundary coordinates
        - x_r, y_r: right boundary coordinates
        - w_l, w_r: left and right track widths
        - heading: heading angles at each centerline point (radians)
        - raceline: optional reference racing line (Nx2 array) or None if unavailable
    """

    track_path = os.path.join(base_path, "tracks", f"{track_name}.csv")
    raceline_path = os.path.join(base_path, "racelines", f"{track_name}.csv")

    if not os.path.exists(track_path):
        raise FileNotFoundError(f"Track file not found: {track_path}")

    # Load track centerline and widths
    df = pd.read_csv(track_path, comment='#', header=None,
                     names=["x_m", "y_m", "w_tr_right_m", "w_tr_left_m"])

    x_c = df["x_m"].to_numpy()
    y_c = df["y_m"].to_numpy()
    w_r = df["w_tr_right_m"].to_numpy()
    w_l = df["w_tr_left_m"].to_numpy()

    # Compute heading between consecutive centerline points
    dx = np.roll(x_c, -1) - x_c
    dy = np.roll(y_c, -1) - y_c
    heading = np.arctan2(dy, dx)

    # Calculate boundary coordinates by shifting centerline points perpendicular to heading
    heading_left = heading + np.pi / 2
    heading_right = heading - np.pi / 2
    x_l = x_c + w_l * np.cos(heading_left)
    y_l = y_c + w_l * np.sin(heading_left)
    x_r = x_c + w_r * np.cos(heading_right)
    y_r = y_c + w_r * np.sin(heading_right)

    # Attempt to load optional reference racing line
    raceline = None
    if os.path.exists(raceline_path):
        try:
            df_race = pd.read_csv(raceline_path, comment='#', header=None, names=["x_m", "y_m"])
            x_ref = df_race["x_m"].to_numpy()
            y_ref = df_race["y_m"].to_numpy()
            raceline = np.column_stack((x_ref, y_ref))
        except Exception as e:
            print(f"Warning: Failed to load raceline for {track_name}: {e}")

    return {
        "x_c": x_c,
        "y_c": y_c,
        "x_l": x_l,
        "y_l": y_l,
        "x_r": x_r,
        "y_r": y_r,
        "w_l": w_l,
        "w_r": w_r,
        "heading": heading,
        "raceline": raceline,
    }


def build_track_polygon(track):
    """
    Build a closed polygon representing the track boundaries.

    Parameters:
    - track: dict
        Track dictionary as returned by load_track()

    Returns:
    - shapely.geometry.Polygon object representing the drivable area of the track
    """

    left_boundary = np.column_stack((track["x_l"], track["y_l"]))
    right_boundary = np.column_stack((track["x_r"], track["y_r"]))[::-1]  # Reverse right boundary to close polygon properly

    # Create linear rings to ensure boundaries are closed loops
    left_ring = LinearRing(left_boundary)
    right_ring = LinearRing(right_boundary)

    # Combine boundary coordinates to form polygon
    polygon_points = np.vstack((left_ring.coords, right_ring.coords))

    poly = Polygon(polygon_points)

    # Fix invalid polygons by buffering with zero-width (often fixes self-intersections)
    if not poly.is_valid:
        poly = poly.buffer(0)
        if not poly.is_valid:
            raise ValueError("Track polygon is invalid and could not be fixed.")

    return poly
