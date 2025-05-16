# AE4350 – F1 Qualifying Lap Optimisation Using Evolutionary Learning

## 🚗 Project Summary

The goal is to evolve a neural network controller that drives a car around various F1 tracks as fast as possible — simulating a qualifying lap. Instead of using gradient-based learning, the controller is trained using an evolutionary algorithm. The final policy is tested on held-out tracks to assess robustness.

## 🧠 Key Features

- **Neuroevolution** of a small neural network for steering and throttle control
- **Training on multiple real-world tracks** from the [TUMFTM racetrack-database](https://github.com/TUMFTM/racetrack-database)
- **Fitness function** combining lap time with control smoothness and crash penalties
- **Track generalization tests** on unseen circuits
- **Visual comparison** between evolved racelines and reference minimum-curvature racing lines provided by the [TUMFTM racetrack-database](https://github.com/TUMFTM/racetrack-database)

## 🏁 Getting Started

### 1. Clone This Repo

Using HTTPS: https://github.com/FranzZeisler/AE4350.git
Using SSH: git@github.com:FranzZeisler/AE4350.git
Using GitHub CLI: gh repo clone FranzZeisler/AE4350

### 2. Install Dependencies

pip install -r requirements.txt

### 3. How it works

- Each agent is defined by a genome that maps directly to a neural network’s weights.
- At each timestep, the agent receives a 17-dimensional input vector (including distances to walls, heading error, curvature, and speed).
- The network outputs steering and throttle values, which are used to update the car's velocity and heading.
- Agents are evaluated over multiple tracks using a fitness function combining:

  - Lap time (minimized)
  - Smoothness penalty (based on jerkiness of controls)
  - Crash penalty (fixed high cost if the car leaves the track)

- The best-performing agents are selected for reproduction, where they undergo crossover and mutation to create a new generation.
- This process is repeated for a set number of generations or until convergence.
- The final evolved controller is tested on both seen and unseen tracks to evaluate performance and generalization.

### 4. Author

This project was developed as part of the AE4350 – Bio-Inspired Intelligence for Aerospace course at TU Delft.

- Author: Franz Zeisler
- Supervisor: Dr. G.C.H.E. de Croon & Dr.ir. E. van Kampen
- Institution: TU Delft, Faculty of Aerospace Engineering
