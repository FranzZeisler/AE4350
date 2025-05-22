import gym
import numpy as np
from car import F1Car  # Import the F1Car class from your existing file
from track import Track  # Import the Track class from your existing file
from features import extract_features  # Importing the feature extraction function
import matplotlib.pyplot as plt
from shapely.geometry import LinearRing


class RaceTrackEnv(gym.Env):
    def __init__(self, track_name, max_steps=1000, gamma=0.95, dt=0.01):
        """
        Initialize the RaceTrack environment with progress-based reward.
        
        :param track_name: Name of the track to load (from your dataset)
        :param max_steps: Maximum steps before the episode ends
        :param gamma: Discount factor for future rewards
        :param dt: Time step duration (seconds)
        """
        super(RaceTrackEnv, self).__init__()

        # Initialize the track and car
        self.track = Track(track_name)
        self.car = F1Car(self.track.x_c[0], self.track.y_c[0], self.track.heading[0], dt=dt)  # Place car at the start and set time step
        self.max_steps = max_steps
        self.gamma = gamma  # Discount factor for future rewards
        self.dt = dt  # Time step for each update
        self.current_step = 0
        
        # Initialize lists for storing positions and speeds for plotting
        self.positions = [self.car.pos]
        self.speeds = [self.car.speed]
        
        self.crash_point = None  # Initialize crash point to None
        self.previous_progress = 0  # Store progress from previous step
        
        self.steps_to_complete_lap = 0  # To count steps until the lap is completed
        self.lap_time = None  # To store the lap time once a lap is completed
        
        # Action space: steer and throttle (steer from -1 to 1, throttle from -1 to 1)
        self.action_space = gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

        # Observation space: Feature vector based on car and track state
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(extract_features(self.car, self.track)),), dtype=np.float32
        )

    def reset(self):
        """Reset the environment to the initial state."""
        self.car = F1Car(self.track.x_c[0], self.track.y_c[0], self.track.heading[0], self.dt)  # Place car at the start and set time step
        self.current_step = 0
        self.positions = [self.car.pos]  # Reset positions list
        self.speeds = [self.car.speed]   # Reset speeds list
        self.crash_point = None  # Reset crash point
        self.previous_progress = 0  # Reset progress
        self.steps_to_complete_lap = 0  # Reset steps
        self.lap_time = None  # Reset lap time
        return extract_features(self.car, self.track)  # Return the initial state as a feature vector

    def step(self, action):
        """Apply the action to the car and update the environment."""
        steer, throttle = action
        print(f"Action: steer={steer}, throttle={throttle}")
        
        # Update the car's state based on the action
        self.car.update(steer, throttle, self.track)
        
        # Store the car's position and speed after each step
        self.positions.append(self.car.pos)
        self.speeds.append(self.car.speed)
        
        # Check if the car has gone off track (crash)
        if self.track.is_off_track(self.car.pos[0], self.car.pos[1]):
            self.crash_point = self.car.pos  # Store crash point
            done = True
            reward = -10  # Penalize the car for going off-track
        else:
            # Check if the car has completed the lap
            current_progress = self.track.project(self.car.pos[0], self.car.pos[1])
            if current_progress >= 1.0 and self.lap_time is None:  # Lap completion check
                self.lap_time = self.dt * self.steps_to_complete_lap  # Lap time = dt * number of steps
                print(f"Lap completed! Lap time: {self.lap_time:.2f} seconds.")
                done = True  # End the lap
                reward = 100  # High reward for completing the lap
            elif self.current_step >= self.max_steps:
                done = True
                reward = 0  # Neutral reward if max steps are reached
            else:
                done = False
                reward = self.calculate_reward()  # Calculate reward based on car's performance
        
        # Extract features for the next state
        state = extract_features(self.car, self.track)
        
        # Update step count and previous progress
        self.previous_progress = self.track.project(self.car.pos[0], self.car.pos[1])
        self.steps_to_complete_lap += 1  # Increment steps
        self.current_step += 1

        return state, reward, done, {}

    def calculate_reward(self):
        """
        Calculate the proxy reward based on course progress using exponential discounting for future rewards.
        
        :return: A calculated reward value
        """
        # Get normalized progress at this time step
        current_progress = self.track.project(self.car.pos[0], self.car.pos[1])
        
        # Calculate the progress difference (how much progress the car has made since the last step)
        progress_diff = current_progress - self.previous_progress
        
        
        # Calculate the discounted future rewards
        discounted_reward = 0
        for i in range(1, self.max_steps - self.current_step):
            future_progress = self.track.project(self.car.pos[0], self.car.pos[1])
            discounted_reward += (self.gamma ** i) * (future_progress - current_progress)
                
        return discounted_reward

    def render(self):
        """Render the environment by plotting the track and car's trajectory."""
        # Plot the track and trajectory with updated positions and speeds
        plt.figure(figsize=(10, 8))

        # Close and plot track boundaries using dot notation to access the attributes
        x_l_closed = np.append(self.track.x_l, self.track.x_l[0])
        y_l_closed = np.append(self.track.y_l, self.track.y_l[0])
        x_r_closed = np.append(self.track.x_r, self.track.x_r[0])
        y_r_closed = np.append(self.track.y_r, self.track.y_r[0])

        # Plot the track boundaries
        plt.plot(x_l_closed, y_l_closed, 'r-', label="Track Boundary")
        plt.plot(x_r_closed, y_r_closed, 'r-')

        # Plot trajectory
        positions = np.array(self.positions)
        if self.speeds is not None:
            sc = plt.scatter(positions[:, 0], positions[:, 1], c=self.speeds, cmap='jet', s=5)
            cbar = plt.colorbar(sc)
            cbar.set_label("Speed (m/s)")
        else:
            plt.plot(positions[:, 0], positions[:, 1], 'k-', label="Trajectory")

        # Plot crash point
        if self.crash_point is not None:
            plt.plot(self.crash_point[0], self.crash_point[1], 'rx', markersize=14, label="Crash")

        plt.axis("equal")
        plt.legend()
        plt.title("Track and Car Trajectory")
        plt.show()

