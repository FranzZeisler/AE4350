import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def plot_track_and_trajectory(track, positions, speeds=None, crash_point=None, plot_raceline=False):
    """
    Plot track boundaries, car trajectory, and optionally a raceline.

    :param track: dict from load_track
    :param positions: (N, 2) array of x, y positions
    :param speeds: optional (N,) array of speeds at each position
    :param crash_point: optional (x, y) tuple
    :param plot_raceline: bool, whether to plot the raceline from the track dictionary
    """
    plt.figure(figsize=(10, 8))

    # Close and plot track boundaries
    x_l_closed = np.append(track["x_l"], track["x_l"][0])
    y_l_closed = np.append(track["y_l"], track["y_l"][0])
    x_r_closed = np.append(track["x_r"], track["x_r"][0])
    y_r_closed = np.append(track["y_r"], track["y_r"][0])

    plt.plot(x_l_closed, y_l_closed, 'r-', label="Track Boundary")
    plt.plot(x_r_closed, y_r_closed, 'r-')

    # Plot trajectory
    positions = np.array(positions)
    if speeds is not None:
        sc = plt.scatter(positions[:, 0], positions[:, 1], c=speeds, cmap='jet', s=5)
        cbar = plt.colorbar(sc)
        cbar.set_label("Speed (m/s)")
    else:
        plt.plot(positions[:, 0], positions[:, 1], 'k-', label="Trajectory")

    # Optionally plot raceline from track dict
    if plot_raceline and track.get("raceline") is not None:
        raceline = np.array(track["raceline"])
        plt.plot(raceline[:, 0], raceline[:, 1], 'b--', linewidth=2, label="Raceline")

    # Plot crash point
    if crash_point is not None:
        plt.plot(crash_point[0], crash_point[1], 'rx', markersize=14, label="Crash")

    plt.axis("equal")
    plt.legend()
    plt.title("Track with Trajectory" + (" and Raceline" if plot_raceline else ""))
    plt.show()


def plot_multiple_trajectories(track, trajectories, crash_points=None):
    """
    Plot track boundaries and multiple car trajectories.
    :param track: track dict from load_track
    :param trajectories: list of arrays of shape (N_i,2) containing x,y positions
    :param crash_points: optional list of crash (x,y) or None
    """
    # Close and plot boundaries
    plt.figure(figsize=(10, 8))

    # Close left boundary by adding first point at end
    x_l_closed = np.append(track["x_l"], track["x_l"][0])
    y_l_closed = np.append(track["y_l"], track["y_l"][0])
    plt.plot(x_l_closed, y_l_closed, 'r-', label="Track Boundary")

    # Close right boundary similarly, **same color, no label**
    x_r_closed = np.append(track["x_r"], track["x_r"][0])
    y_r_closed = np.append(track["y_r"], track["y_r"][0])
    plt.plot(x_r_closed, y_r_closed, 'r-')
    
    # Plot each trajectory
    for idx, traj in enumerate(trajectories):
        traj = np.array(traj)
        plt.plot(traj[:,0], traj[:,1], linewidth=1, alpha=0.7, label=f"Run {idx+1}")
    
    # Plot crash points if provided
    if crash_points is not None:
        for cp in crash_points:
            if cp is not None:
                plt.plot(cp[0], cp[1], 'kx', markersize=8)
    
    plt.axis("equal")
    plt.legend()
    plt.title("Agent Trajectories Over Multiple Test Runs")
    plt.show()