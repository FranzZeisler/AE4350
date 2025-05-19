import numpy as np
import matplotlib.pyplot as plt

def plot_track_and_trajectory(track, positions, speeds=None, crash_point=None):
    """
    Plot track boundaries and car trajectory.
    :param track: track dict from load_track
    :param positions: array of shape (N,2) containing x,y positions
    :param speeds: optional array of shape (N,) containing speed at each position
    :param crash_point: optional (x,y) tuple for crash point
    """
    plt.figure(figsize=(10, 8))

    # Close left boundary by adding first point at end
    x_l_closed = np.append(track["x_l"], track["x_l"][0])
    y_l_closed = np.append(track["y_l"], track["y_l"][0])
    plt.plot(x_l_closed, y_l_closed, 'r-', label="Track Boundary")

    # Close right boundary similarly, **same color, no label**
    x_r_closed = np.append(track["x_r"], track["x_r"][0])
    y_r_closed = np.append(track["y_r"], track["y_r"][0])
    plt.plot(x_r_closed, y_r_closed, 'r-')

    # Plot the trajectory
    positions = np.array(positions)
    if speeds is not None:
        sc = plt.scatter(positions[:, 0], positions[:, 1], c=speeds, cmap='jet', s=5)
        cbar = plt.colorbar(sc)
        cbar.set_label("Speed (m/s)")
    else:
        plt.plot(positions[:, 0], positions[:, 1], 'k-', label="Trajectory")

    # Plot crash point if provided
    if crash_point is not None:
        plt.plot(crash_point[0], crash_point[1], 'rx', markersize=14, label="Crash ❌")

    plt.axis("equal")
    plt.legend()
    plt.title("Track with trajectory")
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