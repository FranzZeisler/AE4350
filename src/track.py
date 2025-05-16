import pandas as pd 
import numpy as np 
import os

def load_track(track_name, base_path="data"):
    """
    Load and process a racetrack from TUMFTM dataset.

    Parameters:
    - track_name: str — name of the CSV file (without .csv)
    - base_path: str — path to 'data' folder

    Returns:
    - Dictionary with centerline, headings, boundaries, widths, and raceline (optional)
    """

    track_path = os.path.join(base_path, "tracks", f"{track_name}.csv")
    race_line_path = os.path.join(base_path, "racelines", f"{track_name}.csv")

    # === Load centerline & width data ===
    df = pd.read_csv(track_path, comment='#', header=None, names=["x_m", "y_m", "w_tr_right_m", "w_tr_left_m"])
    x_c = df["x_m"].to_numpy()
    y_c = df["y_m"].to_numpy()
    w_r = df["w_tr_right_m"].to_numpy()
    w_l = df["w_tr_left_m"].to_numpy()

    # === Compute heading to next point ===
    dx = np.roll(x_c, -1) - x_c
    dy = np.roll(y_c, -1) - y_c
    hdg = np.arctan2(dy, dx)

    # === Compute left/right boundaries ===
    hdg_left = hdg + np.pi / 2
    hdg_right = hdg - np.pi / 2
    x_l = x_c + w_l * np.cos(hdg_left)
    y_l = y_c + w_l * np.sin(hdg_left)
    x_r = x_c + w_r * np.cos(hdg_right)
    y_r = y_c + w_r * np.sin(hdg_right)

    # === Optionally load reference raceline ===
    try:
        df_raceline = pd.read_csv(race_line_path, comment='#', header=None, names=["x_m", "y_m"])
        x_ref = df_raceline["x_m"].to_numpy()
        y_ref = df_raceline["y_m"].to_numpy()
        raceline = np.stack((x_ref, y_ref), axis=1)
    except FileNotFoundError:
        raceline = None

    # === Package as dictionary ===
    return {
        "x_c": x_c,
        "y_c": y_c,
        "x_l": x_l,
        "y_l": y_l,
        "x_r": x_r,
        "y_r": y_r,
        "w_l": w_l,
        "w_r": w_r,
        "heading": hdg,
        "raceline": raceline,
    }

