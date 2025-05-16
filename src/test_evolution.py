from track_load import load_track
from evolution import evolutionary_algorithm
import matplotlib.pyplot as plt

def main():
    track = load_track("Austin")

    best_genome, best_genomes, crash_flags = evolutionary_algorithm(track, population_size=15, generations=10)
    print("Best genome found:", best_genome)

    # Optionally: After evolution finishes, plot best genome lap again to inspect
    # from simulation import run_simulation
    # lap_trace, crashed, finished, lap_time = run_simulation(best_genome, track, spd_init=5.0, dt=0.01, max_time=300)
    # if lap_trace is not None and len(lap_trace) > 0:
    #     x = lap_trace[:, 0]
    #     y = lap_trace[:, 1]
    #     plt.figure(figsize=(10,8))
    #     plt.plot(track["x_c"], track["y_c"], 'orange', label='Centerline')
    #     plt.plot(track["x_l"], track["y_l"], 'yellow', label='Left Boundary')
    #     plt.plot(track["x_r"], track["y_r"], 'teal', label='Right Boundary')
    #     plt.plot(x, y, 'r-', linewidth=2, label='Best Genome Lap')
    #     plt.title('Best Genome Final Lap Trajectory')
    #     plt.xlabel('X [m]')
    #     plt.ylabel('Y [m]')
    #     plt.legend()
    #     plt.axis('equal')
    #     plt.grid(True)
    #     plt.show()

if __name__ == "__main__":
    main()
