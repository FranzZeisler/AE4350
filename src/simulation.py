import numpy as np
from track import load_track

class Track:
    def __init__(self, track_name, base_path="data"):
        self.data = load_track(track_name, base_path)
        self.centerline = np.stack((self.data['x_c'], self.data['y_c']), axis=1)
        self.left_boundary = np.stack((self.data['x_l'], self.data['y_l']), axis=1)
        self.right_boundary = np.stack((self.data['x_r'], self.data['y_r']), axis=1)
        self.heading = self.data["heading"]

    def get_closest_index(self, position):
        dists = np.linalg.norm(self.centerline - position, axis=1)
        return np.argmin(dists)

    def is_off_track(self, position):
        idx = self.get_closest_index(position)
        left = self.left_boundary[idx]
        right = self.right_boundary[idx]
        track_width_vec = left - right
        track_vec_unit = track_width_vec / np.linalg.norm(track_width_vec)
        point_vec = position - right
        proj_length = np.dot(point_vec, track_vec_unit)
        return proj_length < 0 or proj_length > np.linalg.norm(track_width_vec)


class Car:
    def __init__(self, start_position, start_heading=0.0):
        self.position = np.array(start_position, dtype=np.float64)
        self.velocity = 0.0
        self.angle = start_heading  # radians

    def update(self, throttle, steering, dt=0.1, max_steering=np.pi/4, max_throttle=1.0):
        throttle = np.clip(throttle, -1.0, 1.0) * max_throttle
        steering = np.clip(steering, -1.0, 1.0) * max_steering
        self.velocity += throttle * dt
        self.velocity = max(0.0, self.velocity)
        self.angle += steering * dt
        direction = np.array([np.cos(self.angle), np.sin(self.angle)])
        self.position += direction * self.velocity * dt


def run_simulation(agent, track_name="monza"):
    track = Track(track_name)
    car = Car(start_position=track.centerline[0])
    total_reward = 0.0
    max_steps = 500
    dt = 0.1

    for _ in range(max_steps):
        idx = track.get_closest_index(car.position)
        target_heading = track.heading[idx]
        rel_heading = target_heading - car.angle
        rel_heading = np.arctan2(np.sin(rel_heading), np.cos(rel_heading))
        dist_to_center = np.linalg.norm(car.position - track.centerline[idx])
        inputs = np.array([
            car.velocity,
            dist_to_center,
            rel_heading,
        ])
        outputs = agent.forward(inputs)
        throttle, steering = outputs
        car.update(throttle, steering, dt=dt)

        if track.is_off_track(car.position):
            total_reward -= 100
            break

        total_reward += car.velocity * (1 - dist_to_center)

    return total_reward
