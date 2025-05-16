import numpy as np
import random

def initialize_population(pop_size, genome_length, init_std=0.5):
    return [np.random.randn(genome_length) * init_std for _ in range(pop_size)]

def mutate(genome, mutation_std):
    return genome + np.random.randn(*genome.shape) * mutation_std

def crossover(parent1, parent2):
    point = np.random.randint(1, len(parent1))
    child = np.concatenate([parent1[:point], parent2[point:]])
    return child

def select_parents(population, fitnesses, num_parents):
    # Select top-performing genomes
    sorted_indices = np.argsort(fitnesses)[::-1]  # Descending order
    selected = [population[i] for i in sorted_indices[:num_parents]]
    return selected

def evolve_population(population, fitnesses, num_parents, mutation_std):
    parents = select_parents(population, fitnesses, num_parents)
    new_population = []

    while len(new_population) < len(population):
        if len(parents) >= 2:
            p1, p2 = random.sample(parents, 2)
        else:
            p1 = p2 = parents[0]

        child = crossover(p1, p2)
        child = mutate(child, mutation_std)
        new_population.append(child)

    return np.array(new_population)


def run_evolution(
    simulate_function,
    track,
    genome_length,
    pop_size,
    num_parents,
    mutation_std,
    generations,
    render=False,
):
    population = initialize_population(pop_size, genome_length)
    best_fitness = -np.inf
    best_genome = None

    for gen in range(generations):
        fitnesses = []
        for genome in population:
            fitness, lap_time = simulate_function(genome, track, render=render)
            fitnesses.append(fitness)

        fitnesses = np.array(fitnesses)
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_genome = population[max_fitness_idx]

        # Estimate progress % for best genome
        best_fitness_genome = population[max_fitness_idx]
        _, lap_time, best_progress = simulate_function(best_fitness_genome, track, render=False, return_progress=True)
        lap_percent = 100.0 * best_progress
        print(f"Generation {gen + 1} Best Fitness: {best_fitness:.2f} | Lap Completion: {lap_percent:.1f}%")

        population = evolve_population(population, fitnesses, num_parents, mutation_std)

    return best_genome, best_fitness