import gym
from gym import spaces
import numpy as np
from shapely.geometry import Point
from car import Car
from track import load_track, build_track_polygon
from visualisation import plot_track_and_trajectory  # your helper

class RacingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, track_name, dt=0.01):  # Allow dt to be passed in
        '''
        Initialize the Racing Environment.
        Args:
            track_name (str): The name of the track to load.
            dt (float): Time step for the simulation.
        ''' 
        super().__init__()
        self.dt = dt
        self.track   = load_track(track_name)
        self.path    = np.stack((self.track["x_c"], self.track["y_c"]), axis=1)
        self.polygon = build_track_polygon(self.track)

        # Action/Observation spaces
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32) # steer, throttle
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32) # depends on the selected feature vector
        
        # Track completion parameters
        self.max_time = 300.0 # seconds
        self.lap_radius = 5.0 # meters

        # Variables for Rendering
        self.positions = []
        self.speeds = []
        self.crash_point = None

    def reset(self):
        '''
        Reset the environment to an initial state.
        Returns:
            observation (np.ndarray): The observation of the environment.
        '''
        # Reset the car to the start position
        x0, y0 = self.track["x_c"][0], self.track["y_c"][0]
        hdg0    = self.track["heading"][0]
        self.car = Car(x0, y0, hdg0, dt=self.dt)
        self.time = 0.0

        # Reset the render buffers by re-assigning, not slicing
        self.positions = [self.car.pos.copy()]
        self.speeds    = [self.car.speed]
        self.crash_point = None

        # Return the observation
        return self.car.get_feature_vector(self.track, self.path)
    
    def step(self, action):
        '''
        Execute one time step within the environment.
        Args:
            action (np.ndarray): The action taken by the agent.
        Returns:
            observation (np.ndarray): The observation of the environment.
            reward (float): The reward received after taking the action.
            done (bool): Whether the episode has ended.
            info (dict): Additional information about the environment.
        '''
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
            # TODO: Implement the reward function here
            centripetal_penalty = -0.02 * abs(self.car.last_centripetal)   # discourage turning
            accel_bonus         = +1.5 * throttle_agent                    # reward acceleration
            speed_bonus         = +0.1 * self.car.speed                    # encourage speed

            perf_reward = accel_bonus + speed_bonus + centripetal_penalty

        # 7) Return obs, reward, done
        return features, perf_reward, done, info


    def render(self):
        '''
        Renders the environment.
        '''
        plot_track_and_trajectory(
            self.track,
            positions=self.positions,
            speeds=self.speeds,
            crash_point=self.crash_point
        )

    def seed(self, seed=None):
        '''
        Set the random seed for the environment.
        Args:
            seed (int): The random seed to set.
        '''
        np.random.seed(seed)

    def _progress(self, prev_pos, new_pos):
        '''
        Calculate the progress made by the car.
        Args:
            prev_pos (np.ndarray): The previous position of the car.
            new_pos (np.ndarray): The new position of the car.
        Returns:
            int: The progress made by the car.
        '''
        d0 = np.linalg.norm(self.path - prev_pos, axis=1)
        d1 = np.linalg.norm(self.path - new_pos,  axis=1)
        i0 = np.argmin(d0)
        i1 = np.argmin(d1)
        return max(0, i1 - i0)