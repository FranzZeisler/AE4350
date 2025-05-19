import gym
from gym import spaces
import numpy as np
from shapely.geometry import Point
from car import Car
from track import load_track, build_track_polygon
from pursuit_controller import pure_pursuit_control
from visualisation import plot_track_and_trajectory  # your helper

class RacingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, track_name):
        super().__init__()
        self.track   = load_track(track_name)
        self.path    = np.stack((self.track["x_c"], self.track["y_c"]), axis=1)
        self.polygon = build_track_polygon(self.track)
        self.dt      = Car(0,0,0).dt
        
        # action & observation spaces
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        
        # expert parameters
        self.base_lookahead       = 6.8586
        self.lookahead_gain       = 0.1362
        self.alpha                = 0.4005
        self.throttle_threshold_1 = 15.0
        self.throttle_threshold_2 = 10.0
        self.throttle_1 = 1.0
        self.throttle_2 = 0.5171
        self.throttle_3 = 0.6
        
        self.max_time   = 500.0
        self.lap_radius = 5.0

        # for rendering
        self.positions   = []
        self.speeds      = []
        self.crash_point = None
        
    def reset(self):
        x0, y0 = self.track["x_c"][0], self.track["y_c"][0]
        hdg0    = self.track["heading"][0]
        self.car = Car(x0, y0, hdg0)
        self.time = 0.0

        # reset the render buffers by re-assigning, not slicing
        self.positions = [self.car.pos.copy()]
        self.speeds    = [self.car.speed]
        self.crash_point = None

        return self.car.get_feature_vector(self.track, self.path)
    
    def step(self, action):
        # 0) Save previous position
        prev_pos = self.car.pos.copy()

        # 1) Scale agent action to real controls
        steer_agent    = action[0] * self.car.max_steer_rate
        throttle_agent = (action[1] + 1.0) / 2.0  # scale to [0, 1]

        # 2) Update car dynamics
        self.car.update(steer_agent, throttle_agent)
        self.time += self.dt

        # 3) Record for rendering
        self.positions.append(self.car.pos.copy())
        self.speeds.append(self.car.speed)

        # 4) Get observation (feature vector)
        features = self.car.get_feature_vector(self.track, self.path)
        
        # 5) Check termination conditions
        done = False
        info = {}
        if not self.polygon.contains(Point(*self.car.pos)):
            done = True
            perf_reward = -20.0  # light crash penalty
            self.crash_point = self.car.pos.copy()
            info["termination"] = "crash"
        elif self.time > 10.0 and np.linalg.norm(self.car.pos - self.path[0]) < self.lap_radius:
            done = True
            perf_reward = +100.0
            info["termination"] = "lap_complete"
        elif self.time >= self.max_time:
            done = True
            perf_reward = 0.0
            info["termination"] = "timeout"
        else:
            # 6) Reward for driving straight and fast
            centripetal_penalty = -0.02 * abs(self.car.last_centripetal)   # discourage turning
            accel_bonus         = +1.5 * throttle_agent                    # reward acceleration
            speed_bonus         = +0.1 * self.car.speed                    # encourage speed

            perf_reward = accel_bonus + speed_bonus + centripetal_penalty

        # 7) Return obs, reward, done
        return features, perf_reward, done, info



    def render(self, mode='human'):
        # plot with your helper
        plot_track_and_trajectory(
            self.track,
            positions=self.positions,
            speeds=self.speeds,
            crash_point=self.crash_point
        )

    def seed(self, seed=None):
        np.random.seed(seed)

    def _progress(self, prev_pos, new_pos):
        # reward = increase in closest‐point index
        d0 = np.linalg.norm(self.path - prev_pos, axis=1)
        d1 = np.linalg.norm(self.path - new_pos,   axis=1)
        i0 = np.argmin(d0)
        i1 = np.argmin(d1)
        return max(0, i1 - i0)