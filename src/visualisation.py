import numpy as np
import matplotlib.pyplot as plt

def plot_track_and_trajectory(track, positions, speeds=None, crash_point=None):
    plt.figure(figsize=(10, 8))

    # Close left boundary by adding first point at end
    x_l_closed = np.append(track["x_l"], track["x_l"][0])
    y_l_closed = np.append(track["y_l"], track["y_l"][0])
    plt.plot(x_l_closed, y_l_closed, 'r-', label="Track Boundary")

    # Close right boundary similarly, **same color, no label**
    x_r_closed = np.append(track["x_r"], track["x_r"][0])
    y_r_closed = np.append(track["y_r"], track["y_r"][0])
    plt.plot(x_r_closed, y_r_closed, 'r-')

    positions = np.array(positions)
    if speeds is not None:
        sc = plt.scatter(positions[:, 0], positions[:, 1], c=speeds, cmap='jet', s=5)
        cbar = plt.colorbar(sc)
        cbar.set_label("Speed (m/s)")
    else:
        plt.plot(positions[:, 0], positions[:, 1], 'k-', label="Trajectory")

    if crash_point is not None:
        plt.plot(crash_point[0], crash_point[1], 'rx', markersize=14, label="Crash ❌")

    plt.axis("equal")
    plt.legend()
    plt.title("Track with trajectory")
    plt.show()
