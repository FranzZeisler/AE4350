import numpy as np
from features import extract_features  # Importing the external feature extraction function

class F1Car:
    def __init__(self, x, y, heading, dt=0.01):
        """
        Initialize the F1 car with position, heading, and other parameters.
        
        Vehicle-Specific Parameters:
        :param x: Initial x position (meters)
        :param y: Initial y position (meters)
        :param heading: Initial heading (radians)
        :param dt: Time step for the simulation (seconds)
        
        Variables (dynamic states):
        :param dt: Time step for simulation, controlling update frequency
        """
        # Vehicle-Specific Parameters (constants for the vehicle's physical characteristics)
        self.wheelbase = 3.7          # meters (distance between front and rear axles, typical for F1)
        self.max_steering_angle = np.deg2rad(30)  # max steering angle in radians (typical for cars)
        self.max_accel = 15.0         # m/s² (max acceleration for F1-like car)
        self.max_speed = 100.0        # m/s (max speed, representing the top speed of the car)

        # Initial states and variables (changing with time)
        self.pos = np.array([x, y], dtype=float)  # Position in meters (x, y)
        self.heading = heading        # Heading in radians
        self.speed = 0.0              # m/s (initial speed)
        self.previous_speed = 0.0     # m/s (previous speed for acceleration calculation)
        self.steering_angle = 0.0    # Steering angle in radians
        self.throttle = 0.0             # Throttle input (0 to 1)
        self.past_steering_angle = 0.0  # Previous steering angle
        self.past_throttle = 0.0        # Previous throttle input
        self.dt = dt  # Time step for the simulation
        self.velocity = np.array([0.0, 0.0])  # Initial velocity vector in world frame
        self.linear_acceleration = 0.0  # Initial linear acceleration (m/s²)
        self.track_progress = 0.0  # Initialize track progress (0 to 1, normalized)

    def update(self, steer, throttle, track):
        """
        Update the F1 car's position, speed, and heading based on steering and throttle inputs.
        
        :param steer: Steering input (-1 to 1), affecting the steering angle
        :param throttle: Throttle input (-1 to 1), affecting acceleration
        :param track: The track object used to compute the car's progress
        """
        # Clamp the inputs
        self.past_steering_angle = self.steering_angle
        self.past_throttle = self.throttle
        self.steering_angle = np.clip(steer, -self.max_steering_angle, self.max_steering_angle)
        self.throttle = np.clip(throttle, -1, 1)

        # Update speed using throttle input
        self.linear_acceleration = self.throttle * self.max_accel
        self.previous_speed = self.speed  # Store previous speed for acceleration calculation
        self.speed = np.clip(self.speed + self.linear_acceleration * self.dt, 0, self.max_speed)

        # Compute angular velocity from steering and speed (kinematic bicycle model)
        if abs(self.steering_angle) > 1e-4:
            # Calculate the turning radius using the kinematic bicycle model
            turning_radius = self.wheelbase / np.tan(self.steering_angle)

            # Angular velocity (how fast the car is turning)
            angular_velocity = self.speed / turning_radius
        else:
            angular_velocity = 0.0

        # Update heading based on angular velocity and time step
        self.heading += angular_velocity * self.dt

        # Update velocity vector in the world frame based on speed and heading
        self.velocity = np.array([self.speed * np.cos(self.heading),
                                  self.speed * np.sin(self.heading)])

        # Update position based on velocity
        self.pos += self.velocity * self.dt

        # Update track progress (normalized between 0 and 1)
        self.track_progress = track.project(self.pos[0], self.pos[1])  # Assuming track has a project() method

    def get_feature_vector(self, track):
        """
        Extract features for the car's state and return them as a vector.
        :param track: The track object used to compute errors.
        :return: A vector of features for input to the policy network.
        """
        return extract_features(self, track)
