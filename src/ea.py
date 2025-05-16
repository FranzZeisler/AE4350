import numpy as np
import copy

class Evolution:
    def __init__(self, agent_class, population_size, input_size, hidden_size, output_size):
        self.agent_class = agent_class
        self.population = [
            agent_class(input_size, hidden_size, output_size)
            for _ in range(population_size)
        ]

    def evaluate(self, simulation_fn):
        return [simulation_fn(agent) for agent in self.population]

    def select_and_reproduce(self, scores):
        # Fix: sort by score only
        scored_agents = list(zip(scores, self.population))
        scored_agents.sort(key=lambda x: x[0], reverse=True)

        survivors = [agent for _, agent in scored_agents[:len(self.population) // 2]]
        new_population = []

        for _ in range(len(self.population)):
            parent = np.random.choice(survivors)
            child = copy.deepcopy(parent)
            self.mutate(child)
            new_population.append(child)

        self.population = new_population


    def mutate(self, agent, rate=0.1):
        weights = agent.get_weights()
        for k in weights:
            weights[k] += np.random.randn(*weights[k].shape) * rate
        agent.set_weights(weights)
