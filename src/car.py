import numpy as np

from features import extract_features


class Car:
    def __init__(self, x, y, heading, dt=0.01):
        """
        Initialize the car with position, heading, and other parameters.
        :param x: Initial x position (meters)
        :param y: Initial y position (meters)
        :param heading: Initial heading (radians)
        :param dt: Time step for the simulation (seconds)
        """
        self.pos = np.array([x, y], dtype=float)
        self.heading = heading        # radians
        self.speed = 0.0              # m/s

        self.steering_angle = 0.0    # radians
        self.prev_steering_angle = 0.0
        self.throttle = 0.0          # normalized throttle input (-1 to 1)

        self.wheelbase = 3.7          # meters
        self.max_steering_angle = np.deg2rad(30)  # max steering angle

        self.max_accel = 15.0                     # m/s²
        self.max_speed = 100.0                    # m/s
        self.max_lateral_accel = 4.0 * 9.81       # m/s²

        self.dt = dt
        self.velocity = np.array([0.0, 0.0])

    def update(self, steer, throttle):
        """
        Update the car's position and speed based on steering and throttle inputs.
        :param steer: Steering input (-1 to 1)
        :param throttle: Throttle input (-1 to 1)
        """
        # Clamp inputs
        self.prev_steering_angle = self.steering_angle
        self.steering_angle = np.clip(steer, -self.max_steering_angle, self.max_steering_angle)
        throttle = np.clip(throttle, -1, 1)
        self.throttle = throttle

        # Update speed with simple acceleration model
        accel = throttle * self.max_accel
        self.speed = np.clip(self.speed + accel * self.dt, 0, self.max_speed)

        # Compute angular velocity from steering and speed (kinematic bicycle)
        if abs(self.steering_angle) > 1e-4:
            turning_radius = self.wheelbase / np.tan(self.steering_angle)
            max_speed_on_turn = np.sqrt(self.max_lateral_accel * abs(turning_radius))
            if self.speed > max_speed_on_turn:
                self.speed = max_speed_on_turn
            angular_velocity = self.speed / turning_radius
        else:
            angular_velocity = 0.0

        # Update heading and position
        self.heading += angular_velocity * self.dt
        
        # Update velocity vector in world frame
        self.velocity = np.array([
            self.speed * np.cos(self.heading),
            self.speed * np.sin(self.heading)
        ])

        # Update position
        self.pos += self.velocity * self.dt

    def get_feature_vector(self, track):
        """
        Extract features from the car's state and the track.
        :param track: The track data.
        :return: Feature vector.
        """
        return extract_features(self, track)
