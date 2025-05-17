import numpy as np
from features import extract_features 

class Car:
    def __init__(self, x, y, heading):
        """
        Initialize the car's position, heading, speed, and other parameters.
        :param x: Initial x-coordinate of the car.
        :param y: Initial y-coordinate of the car.
        :param heading: Initial heading of the car in radians.
        """
        self.pos = np.array([x, y], dtype=float)
        self.heading = heading  # radians
        self.speed = 10.0        # m/s
        self.velocity = np.array([0.0, 0.0])
        self.wheelbase = 3.6     # meters

        self.max_accel = 15.0       # m/s²
        self.max_decel = 30.0       # m/s²
        self.max_lateral_g = 4.0 * 9.81
        self.top_speed = 70.0       # m/s
        self.max_steer_rate = np.deg2rad(2.0)  # radians per step
        self.steering_angle = 0.0   # radians
        self.dt = 0.1               # seconds

        self.last_lat_accel = 0.0
        self.last_centripetal = 0.0
        self.last_vel_heading = 0.0

    def update(self, target_steer, throttle):
        """
        Update the car's position, speed, and steering angle based on the target steering and throttle.
        :param target_steer: Target steering angle in radians.
        :param throttle: Throttle input, where 1.0 is full throttle and -1.0 is full brake.
        """

        # Update steering angle with rate limit
        steer_diff = np.clip(target_steer - self.steering_angle, -self.max_steer_rate, self.max_steer_rate)
        self.steering_angle += steer_diff

        # Update speed based on throttle input
        # Positive throttle for acceleration, negative for deceleration
        if throttle >= 0:
            accel = throttle * self.max_accel
        else:
            accel = throttle * self.max_decel

        # Apply acceleration to speed, ensuring it doesn't exceed top speed or go below zero
        self.speed += accel * self.dt
        self.speed = np.clip(self.speed, 0, self.top_speed)

        # Calculate lateral acceleration and adjust speed if necessary
        turning_radius = self.wheelbase / (np.tan(self.steering_angle) + 1e-6)
        lat_accel = self.speed ** 2 / turning_radius
        if abs(lat_accel) > self.max_lateral_g:
            self.speed = np.sqrt(abs(self.max_lateral_g * turning_radius))

        # Update position and heading based on speed and steering angle
        # Calculate angular velocity and update heading 
        angular_velocity = self.speed / self.wheelbase * np.tan(self.steering_angle)
        self.heading += angular_velocity * self.dt

        # Update position based on speed and heading
        dx = self.speed * np.cos(self.heading) * self.dt
        dy = self.speed * np.sin(self.heading) * self.dt
        self.velocity = np.array([dx, dy]) / self.dt
        self.pos += np.array([dx, dy])

        # Store last values for feature extraction
        self.last_lat_accel = lat_accel
        self.last_centripetal = lat_accel
        self.last_vel_heading = self.velocity_heading()

    def velocity_heading(self):
        """
        Calculate the heading of the car based on its velocity vector.
        :return: The heading of the car in radians.
        """
        vel_angle = np.arctan2(self.velocity[1], self.velocity[0])
        angle_diff = vel_angle - self.heading
        # Normalize the angle difference to be within [-pi, pi]
        return (angle_diff + np.pi) % (2 * np.pi) - np.pi

    def get_feature_vector(self, track, path_points):
        """
        Extract features from the car's state and the track.
        :param track: The track data.
        :param path_points: The points along the track centerline.
        :return: A feature vector containing various information about the car and track.
        """
        return extract_features(self, track, path_points)
