import numpy as np
from simulate_neural import simulate_track_neural
from simulate_pid import simulate_track_pid
from simulate_pursuit import simulate_track_pursuit
from track import load_track
from evolve import run_evolution
import matplotlib.pyplot as plt


if __name__ == "__main__":
    track = load_track("SaoPaulo")

    # Neural controller
    genome_size = 674  # Example size
    best_genome, best_fitness, fitness_history = run_evolution(
        simulate_function=simulate_track_neural,
        track=track,
        genome_length=genome_size,
        pop_size=50,
        num_parents=10,
        mutation_std=0.1,
        generations=20,
        render=False
    )

    # Display the fitness history
    plt.plot(fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Fitness Progress Over Generations")
    plt.show()

    # Print the best genome and fitness
    print("Best fitness found:", best_fitness)
    print("Best genome found:", best_genome)
    # Visualize the best genome
    simulate_track_neural(best_genome, track, render=True)


    # PID controller
    # simulate_track_pid(track)

    # Pure pursuit controller
    # simulate_track_pursuit(track)

