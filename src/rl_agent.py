from matplotlib import pyplot as plt
import numpy as np
import gym
from shapely.geometry import Point
from car import Car
from pursuit_controller import pure_pursuit_control
from features import extract_features  # adjust if needed

class CarEnv(gym.Env):
    def __init__(self, track):
        super().__init__()
        self.track = track
        self.path_points = np.stack((track["x_c"], track["y_c"]), axis=1)
        self.track_polygon = self._build_track_polygon(track)

        # Action space: continuous steering (-1 to 1) and throttle (0 to 1)
        # You can normalize later as needed
        self.action_space = gym.spaces.Box(low=np.array([-1, 0]), high=np.array([1, 1]), dtype=np.float32)

        # Observation space: shape matches your feature vector length
        dummy_car = Car(track["x_c"][0], track["y_c"][0], track["heading"][0])
        dummy_features = extract_features(dummy_car, track, self.path_points)
        obs_len = len(dummy_features)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)

        self.car = None
        self.time_elapsed = 0.0
        self.max_time = 500.0
        self.done = False

    def _build_track_polygon(self, track):
        from shapely.geometry import LinearRing, Polygon
        left_boundary = np.column_stack((track["x_l"], track["y_l"]))
        right_boundary = np.column_stack((track["x_r"], track["y_r"]))[::-1]

        left_ring = LinearRing(left_boundary)
        right_ring = LinearRing(right_boundary)

        polygon_points = np.vstack((left_ring.coords, right_ring.coords))
        poly = Polygon(polygon_points)

        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly

    def reset(self):
        self.car = Car(self.track["x_c"][0], self.track["y_c"][0], self.track["heading"][0])
        self.time_elapsed = 0.0
        self.done = False

        # For rendering
        self.positions = [self.car.pos.copy()]
        self.speeds = [self.car.speed]
        self.crash_point = None

        obs = extract_features(self.car, self.track, self.path_points)
        return obs

    def step(self, action):
        if self.done:
            raise RuntimeError("Step called after done=True. Please call reset().")

        max_steer_angle = np.deg2rad(30)
        steering_angle = action[0] * max_steer_angle
        throttle = action[1]

        self.car.update(steering_angle, throttle)
        self.time_elapsed += self.car.dt

        # Store positions and speeds for rendering
        self.positions.append(self.car.pos.copy())
        self.speeds.append(self.car.speed)

        car_point = Point(self.car.pos[0], self.car.pos[1])
        if not self.track_polygon.contains(car_point):
            self.done = True
            reward = -100.0
            self.crash_point = self.car.pos.copy()
            obs = extract_features(self.car, self.track, self.path_points)
            return obs, reward, self.done, {"crash": True}

        dist_to_start = np.linalg.norm(self.car.pos - self.path_points[0])
        if self.time_elapsed > 10.0 and dist_to_start < 5.0:
            self.done = True
            reward = 1000.0
            obs = extract_features(self.car, self.track, self.path_points)
            return obs, reward, self.done, {"lap_completed": True}

        if self.time_elapsed >= self.max_time:
            self.done = True
            reward = -10.0
            obs = extract_features(self.car, self.track, self.path_points)
            return obs, reward, self.done, {"timeout": True}

        closest_idx = np.argmin(np.linalg.norm(self.path_points - self.car.pos, axis=1))
        progress = closest_idx / len(self.path_points)
        reward = progress * 10.0 + self.car.speed * 0.1

        obs = extract_features(self.car, self.track, self.path_points)
        return obs, reward, self.done, {}

    def render(self):
        positions = np.array(self.positions)
        speeds = np.array(self.speeds)

        plt.figure(figsize=(10, 8))
        plt.plot(self.track["x_l"], self.track["y_l"], 'r-', label="Left boundary")
        plt.plot(self.track["x_r"], self.track["y_r"], 'b-', label="Right boundary")

        sc = plt.scatter(positions[:, 0], positions[:, 1], c=speeds, cmap='jet', s=5)
        cbar = plt.colorbar(sc)
        cbar.set_label("Speed (m/s)")

        if self.crash_point is not None:
            plt.plot(self.crash_point[0], self.crash_point[1], 'rx', markersize=14, label="Crash ❌")

        plt.axis("equal")
        plt.legend()
        plt.title("Car Trajectory with Speed Coloring")
        plt.show()