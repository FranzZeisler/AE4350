import gym
from gym import spaces
import numpy as np
from shapely.geometry import Point
from car import Car
from track import load_track, build_track_polygon
from visualisation import plot_track_and_trajectory

class RacingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, track_name):
        super().__init__()
        self.track = load_track(track_name)
        self.car = Car(self.track["x_c"][0], self.track["y_c"][0], self.track["heading"][0])
        self.path_points = np.stack((self.track["x_c"], self.track["y_c"]), axis=1)
        self.track_polygon = build_track_polygon(self.track)
        self.dt = self.car.dt

        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        obs_dim = 4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.time_elapsed = 0.0
        self.max_time = 500.0
        self.lap_radius = 5.0

        self.positions = []  # For rendering trajectory
        self.crash_point = None

    def reset(self):
        self.car = Car(self.track["x_c"][0], self.track["y_c"][0], self.track["heading"][0])
        self.time_elapsed = 0.0
        self.positions = [self.car.pos.copy()]
        self.crash_point = None
        return self.car.get_feature_vector(self.track, self.path_points)

    def step(self, action):
        steer = action[0] * self.car.max_steer_rate
        throttle = (action[1] + 1) / 2  # scale from [-1,1] to [0,1]

        self.car.update(steer, throttle)
        self.time_elapsed += self.dt
        self.positions.append(self.car.pos.copy())

        obs = self.car.get_feature_vector(self.track, self.path_points)

        done = False
        reward = 0.0
        info = {}

        if not self.track_polygon.contains(Point(self.car.pos[0], self.car.pos[1])):
            done = True
            reward = -100.0
            self.crash_point = self.car.pos.copy()
            info["termination_reason"] = "crash"
            return obs, reward, done, info

        if self.time_elapsed > 10.0:
            dist_to_start = np.linalg.norm(self.car.pos - self.path_points[0])
            if dist_to_start < self.lap_radius:
                done = True
                reward = 100.0
                info["termination_reason"] = "lap_complete"
                return obs, reward, done, info

        reward += self.car.speed * self.dt
        reward -= 0.1 * abs(steer)

        if self.time_elapsed >= self.max_time:
            done = True
            info["termination_reason"] = "timeout"

        return obs, reward, done, info

    def render(self, mode='human'):
        import matplotlib.pyplot as plt
        if mode != 'human':
            raise NotImplementedError(f"Render mode {mode} not implemented")

        plot_track_and_trajectory(self.track, self.positions, crash_point=self.crash_point)

    def seed(self, seed=None):
        np.random.seed(seed)
