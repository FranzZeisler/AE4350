# 🏎️ AE4350 – F1 Qualifying Lap Optimisation Using Evolutionary Learning

## 🚗 Project Summary

This project develops an autonomous F1 racing agent that learns to minimise qualifying lap times using a hybrid of **behavioural cloning (BC)** and **reinforcement learning (TD3)**. The agent is trained and evaluated on real-world circuits from the [TUMFTM racetrack-database](https://github.com/TUMFTM/racetrack-database).

## 🧠 Key Features
- Hybrid learning pipeline: Behavioural Cloning → TD3 Reinforcement Learning
- Custom racing environment using a kinematic bicycle model
- Track-specific policy optimisation with reward functions balancing speed and smoothness
- Lap time improvements from **74s (baseline)** to **62s (final agent)**
- **Training on multiple real-world tracks** from the [TUMFTM racetrack-database](https://github.com/TUMFTM/racetrack-database)

## 🏁 Getting Started

### 1. Clone This Repo

- Using HTTPS: https://github.com/FranzZeisler/AE4350.git
- Using SSH: git@github.com:FranzZeisler/AE4350.git
- Using GitHub CLI: gh repo clone FranzZeisler/AE4350

### 2. Install Dependencies

pip install -r requirements.txt

### 3. How it works

The training pipeline consists of four key stages:

1. **Expert Data Generation**  
   A pure pursuit controller drives the car around the circuit to generate expert trajectories. This provides safe and reasonably fast baseline data.

2. **Behavioural Cloning (BC)**  
   A neural network policy is trained to imitate the expert's steering and throttle actions via supervised learning. This gives the agent a strong starting point.

3. **Reinforcement Learning (TD3)**  
   The cloned policy is fine-tuned using Twin Delayed Deep Deterministic Policy Gradient (TD3). The agent interacts with the environment to further reduce lap times and improve control smoothness.

4. **Evaluation**  
   The final policy is tested on the training track and optionally on unseen circuits. Performance is assessed based on lap time and qualitative raceline analysis.

The agent observes key features such as distance to track boundaries, heading error, curvature, and speed. It outputs steering and throttle commands, driving the car in a closed simulation loop.

## ✏️ Author

This project was developed as part of the AE4350 – Bio-Inspired Intelligence for Aerospace course at TU Delft.

- Author: Franz Zeisler
- Supervisor: Dr. G.C.H.E. de Croon & Dr.ir. E. van Kampen
- Institution: TU Delft, Faculty of Aerospace Engineering
