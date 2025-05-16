import numpy as np
import random
import matplotlib.pyplot as plt
from simulation import run_simulation  # Adjust import if needed

def mutate_genome(genome, mutation_rate=0.1, mutation_scale=0.1):
    """Mutate genome with given rate and scale."""
    new_genome = genome.copy()
    for i in range(len(genome)):
        if random.random() < mutation_rate:
            new_genome[i] += random.gauss(0, mutation_scale * abs(genome[i] if genome[i] != 0 else 1))
            # Clip genome parameters to sensible ranges
            if i == 0 or i == 5:
                new_genome[i] = np.clip(new_genome[i], 0, 1)
            elif i == 1:
                new_genome[i] = np.clip(new_genome[i], 0.1, 5)
            elif i == 2:
                new_genome[i] = np.clip(new_genome[i], 1, 90)
            elif i == 3 or i == 4:
                new_genome[i] = int(np.clip(new_genome[i], 1, 30))
    return new_genome

def crossover(genome1, genome2):
    """Single-point crossover between two genomes."""
    point = random.randint(1, len(genome1) - 1)
    child = genome1[:point] + genome2[point:]
    return child

def generate_random_genome():
    """Generate a random genome with conservative parameter ranges."""
    return [
        random.uniform(0.1, 0.5),       # steer_gain
        random.uniform(0.5, 3.0),       # centerline_tol
        random.uniform(5, 30),          # heading_tol
        random.randint(3, 8),           # future_idx_short
        random.randint(10, 20),         # future_idx_long
        random.uniform(0.1, 0.5),       # accel_gain
    ]

def evolutionary_algorithm(track, population_size=15, generations=10):
    """Run evolutionary algorithm to optimize car controller genome."""

    best_genomes_evo = []
    best_crash_status = []

    # Prepare track boundaries for plotting
    x_c = track["x_c"]
    y_c = track["y_c"]
    x_l = track["x_l"]
    y_l = track["y_l"]
    x_r = track["x_r"]
    y_r = track["y_r"]

    # Initialize population
    population = [generate_random_genome() for _ in range(population_size)]

    for gen in range(generations):
        fitnesses = []
        lap_traces = []
        crash_flags = []

        print(f"Generation {gen + 1}/{generations}")

        for i, genome in enumerate(population):
            lap_trace, crashed, finished, lap_time = run_simulation(genome, track, spd_init=5.0, dt=0.01, max_time=300)

            fitness = 9999.0 if crashed or not finished else lap_time

            fitnesses.append(fitness)
            lap_traces.append(lap_trace)
            crash_flags.append(crashed)

            print(f"  Agent {i + 1}: Time = {lap_time:.2f}s, Crashed = {crashed}")

        best_idx = np.argmin(fitnesses)
        best_genome = population[best_idx]
        best_genomes_evo.append(best_genome)
        best_crash_status.append(crash_flags[best_idx])

        # Plot best lap of current generation
        lap_trace = lap_traces[best_idx]
        plt.figure(figsize=(10, 8))
        plt.plot(x_c, y_c, 'orange', linewidth=2, label='Centerline')
        plt.plot(x_l, y_l, 'yellow', linewidth=2, label='Left Boundary')
        plt.plot(x_r, y_r, 'teal', linewidth=2, label='Right Boundary')

        if lap_trace is not None and len(lap_trace) > 0:
            x = lap_trace[:, 0]
            y = lap_trace[:, 1]
            label = f'Gen {gen + 1} {"(Crash)" if crash_flags[best_idx] else ""}'
            plt.plot(x, y, 'r-', linewidth=2, label=label)

        plt.title(f'Best Lap Trajectory - Generation {gen + 1}')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()

        # Create next generation with elitism
        new_population = [best_genome]

        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population, 2)
            child = crossover(parent1, parent2)
            child = mutate_genome(child)
            new_population.append(child)

        population = new_population

    return best_genomes_evo[-1], best_genomes_evo, best_crash_status
