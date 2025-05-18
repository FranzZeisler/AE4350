import numpy as np
from stable_baselines3 import PPO
#from pursuit_optimiser import optimize_params_grid
from rl_agent import CarEnv
from simulate_neural import simulate_track_neural
from simulate_pid import simulate_track_pid
from simulate_pursuit import simulate_track_pursuit
from track import load_track
from evolve import run_evolution
import matplotlib.pyplot as plt


if __name__ == "__main__":
    track = load_track("Austin")

    # Neural controller
    # genome_size = 674  # Example size
    # best_genome, best_fitness, fitness_history = run_evolution(
    #     simulate_function=simulate_track_neural,
    #     track=track,
    #     genome_length=genome_size,
    #     pop_size=30,
    #     num_parents=10,
    #     mutation_std=0.2,
    #     generations=20,
    #     render=False
    # )

    # # Display the fitness history
    # plt.plot(fitness_history)
    # plt.xlabel("Generation")
    # plt.ylabel("Best Fitness")
    # plt.title("Fitness Progress Over Generations")
    # plt.show()

    # # Print the best genome and fitness
    # print("Best fitness found:", best_fitness)
    # print("Best genome found:", best_genome)
    # # Visualize the best genome
    # simulate_track_neural(best_genome, track, render=True)


    # PID controller
    # simulate_track_pid(track)

    # Pure pursuit controller
    # optimize_params_grid(track)
    #simulate_track_pursuit(track, plot_speed=True)

    env = CarEnv(track)

    # Create PPO model with your env
    model = PPO("MlpPolicy", env, verbose=1)

    # Train for 1000 timesteps (adjust as needed)
    model.learn(total_timesteps=10000)

    # Save the model
    model.save("ppo_car_model")

    # Test the trained model
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # or your agent's action
        obs, reward, done, info = env.step(action)

    print(f"Episode done. Rendering...")
    env.render()
