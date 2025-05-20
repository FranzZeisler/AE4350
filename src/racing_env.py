import gym
from gym import spaces
import numpy as np
from shapely.geometry import Point
from car import Car
from track import load_track, build_track_polygon
from visualisation import plot_track_and_trajectory  # your helper

class RacingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, track_name, dt=0.01):
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

        # Precompute cumulative track lengths for progress calculation
        self.cumulative_lengths = np.zeros(len(self.path))
        for i in range(1, len(self.path)):
            self.cumulative_lengths[i] = self.cumulative_lengths[i-1] + np.linalg.norm(self.path[i] - self.path[i-1])
        self.total_length = self.cumulative_lengths[-1]

        # Action/Observation spaces
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)  # steer, throttle
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)  # updated feature vector length

        # Track completion parameters
        self.max_time = 300.0  # seconds
        self.lap_radius = 5.0  # meters

        # Car progress tracking
        self.prev_progress = 0.0  # float progress fraction

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
        self.prev_progress = 0.0  # reset progress

        # Reset the render buffers by re-assigning, not slicing
        self.positions = [self.car.pos.copy()]
        self.speeds    = [self.car.speed]
        self.crash_point = None

        # Return the initial observation
        return self.car.get_feature_vector(self.track)
    
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
        steer_agent    = action[0]
        throttle_agent = action[1]

        # 2) Update car dynamics
        self.car.update(steer_agent, throttle_agent)
        self.time += self.dt

        # 3) Record for rendering
        self.positions.append(self.car.pos.copy())
        self.speeds.append(self.car.speed)

        # 4) Get observation (feature vector)
        features = self.car.get_feature_vector(self.track)
        
        # 5) Check termination conditions
        done = False
        info = {}
        if not self.polygon.contains(Point(*self.car.pos)):
            done = True
            fitness = -20.0  # light crash penalty
            self.crash_point = self.car.pos.copy()
            info["termination"] = "crash"
        elif self.time > 10.0 and np.linalg.norm(self.car.pos - self.path[0]) < self.lap_radius:
            done = True
            fitness = +100.0
            info["termination"] = "lap_complete"
            info["lap_time"] = self.time
        elif self.time >= self.max_time:
            done = True
            fitness = 0.0
            info["termination"] = "timeout"
        else:
            # 6) Update fitness score
            fitness = self.update_fitness(action)

        # 7) Return obs, reward, done, info
        return features, fitness, done, info
    
    def update_fitness(self, action):
        '''
        Update the fitness score based on the car's performance.
        Args:
            action (np.ndarray): The action taken by the agent.
        Returns:
            fitness (float): The updated fitness score.
        '''
        # Progress reward
        progress_delta = self.compute_progress()
        progress_bonus = progress_delta * 20.0  # tune scaling factor

        # Throttle reward
        accel_bonus = +1.5 * action[1]

        # Speed reward
        speed_norm = self.car.speed / self.car.max_speed
        speed_bonus = +0.1 * speed_norm

        return progress_bonus + accel_bonus + speed_bonus 

    def compute_progress(self):
        """
        Computes the percentage of progress made along the track centerline (0.0 to 1.0).
        Returns:
            progress_delta (float): Progress since last step as fraction of total track length.
        """
        # Find closest point on path
        dists = np.linalg.norm(self.path - self.car.pos, axis=1)
        idx = np.argmin(dists)
        
        # Get cumulative length at closest point
        current_length = self.cumulative_lengths[idx]
        current_progress = current_length / self.total_length
        
        prev_progress = self.prev_progress
        
        self.prev_progress = current_progress
        
        return current_progress - prev_progress

    def render(self):
        '''
        Renders the environment.
        '''
        plot_track_and_trajectory(
            self.track,
            positions=self.positions,
            speeds=self.speeds,
            crash_point=self.crash_point,
            plot_raceline=True
        )

    def seed(self, seed=None):
        '''
        Set the random seed for the environment.
        Args:
            seed (int): The random seed to set.
        '''
        np.random.seed(seed)
