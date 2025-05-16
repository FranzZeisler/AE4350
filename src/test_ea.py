from ea import (
    initialize_population,
    evaluate_population,
    next_generation
)
from simulation import simulate_agent
import matplotlib.pyplot as plt

# EA parameters
population_size = 20
genome_size = 674
generations = 10
n_elites = 5
mutation_rate = 0.1
mutation_strength = 0.2

# Initialize
population = initialize_population(population_size, genome_size)

# Evolution loop
for gen in range(generations):
    fitnesses = evaluate_population(population, simulate_agent)
    print(f"Gen {gen:02d} | Best fitness: {fitnesses.max():.2f}")
    population = next_generation(population, fitnesses, n_elites, mutation_rate, mutation_strength)

# Optional: plot or save best genome
best_idx = fitnesses.argmax()
best_genome = population[best_idx]
print("Best genome found:", best_genome)
