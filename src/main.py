from ea import Evolution
from nn_agent import NNAgent
from simulation import run_simulation

def main():
    evo = Evolution(
        agent_class=NNAgent,
        population_size=20,
        input_size=3,
        hidden_size=5,
        output_size=2
    )

    for generation in range(100):
        scores = evo.evaluate(lambda agent: run_simulation(agent, track_name="Austin"))
        best_score = max(scores)
        print(f"Generation {generation}: Best Score = {best_score:.2f}")
        evo.select_and_reproduce(scores)

if __name__ == "__main__":
    main()
