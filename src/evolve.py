import numpy as np
from controller import SimpleNNController
from simulate import simulate

def evolve(track):
    pop_size = 10
    gens = 20
    genomes = np.random.randn(pop_size, 4)  # 2x2 weights

    for gen in range(gens):
        scores = []
        for genome in genomes:
            ctrl = SimpleNNController(genome)
            fitness = simulate(ctrl, track)
            scores.append(fitness)

        best_idx = np.argmin(scores)
        best = genomes[best_idx]
        print(f"Gen {gen}: Best fitness = {scores[best_idx]:.2f}")

        # Selection: take top 2, make offspring with mutation
        new_genomes = [best.copy()]
        for _ in range(pop_size - 1):
            offspring = best + np.random.randn(4) * 0.1
            new_genomes.append(offspring)

        genomes = np.array(new_genomes)
