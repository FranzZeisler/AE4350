import gym
from gym import spaces
import numpy as np
from shapely.geometry import Point, LineString
from car import Car
from track import load_track, build_track_polygon
from visualisation import plot_track_and_trajectory  # your helper

# === Tunable global parameters ===

MAX_TIME = 300.0       # Max episode duration (seconds)
LAP_RADIUS = 5.0       # Radius to detect lap completion (meters)
MIN_LAP_TIME = 10.0    # Minimum time before lap can be considered complete (seconds)

class RacingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, track_name, dt):
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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)  # feature vector length

        # Track completion parameters
        self.max_time = MAX_TIME
        self.lap_radius = LAP_RADIUS
        self.min_lap_time = MIN_LAP_TIME

        # Car progress tracking
        self.prev_progress = 0.0  # float progress fraction
        self.steps = 0

        # Variables for Rendering
        self.positions = []
        self.speeds = []
        self.crash_point = None

        # Define finish line as a small line segment perpendicular to track start direction
        self.define_finish_line()
        self.crossed_finish_line = False  # Flag to detect crossing only once per lap

    def define_finish_line(self, line_length=20.0):
        """
        Define the finish line as a small line segment perpendicular to the track start direction.
        Args:
            line_length (float): Length of the finish line segment in meters.
        """
        start_point = self.path[0]
        direction_vector = self.path[1] - self.path[0]
        perp_vector = np.array([-direction_vector[1], direction_vector[0]])
        perp_vector /= np.linalg.norm(perp_vector)  # normalize

        p1 = start_point + perp_vector * (line_length / 2)
        p2 = start_point - perp_vector * (line_length / 2)
        self.finish_line = LineString([p1, p2])

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
        self.prev_progress = 0.0
        self.steps = 0

        # Reset the render buffers
        self.positions = [self.car.pos.copy()]
        self.speeds    = [self.car.speed]
        self.crash_point = None

        self.crossed_finish_line = False  # Reset finish line crossing flag on reset

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
        steer_agent, throttle_agent = action

        # Update car dynamics
        self.car.update(steer_agent, throttle_agent)
        self.time += self.dt

        # Record for rendering
        self.positions.append(self.car.pos.copy())
        self.speeds.append(self.car.speed)

        # Get observation (feature vector)
        features = self.car.get_feature_vector(self.track)

        # Check termination conditions
        done = False
        info = {}

        if not self.polygon.contains(Point(*self.car.pos)):
            done = True
            info["termination"] = "crash"
            self.crash_point = self.car.pos.copy()
        elif self.check_lap_complete():
            done = True
            info["termination"] = "lap_complete"
            info["lap_time"] = self.format_lap_time(self.time)
            print(f"Lap time: {info['lap_time']}")
        elif self.time >= self.max_time:
            done = True
            info["termination"] = "timeout"

        # Calculate reward
        #reward = self.update_fitness(action, done, info)
        reward = self.update_fitness(action, info)

        #Update number steps
        self.steps += 1

        return features, reward, done, info

    def format_lap_time(self, time_seconds):
        minutes = int(time_seconds // 60)
        seconds = int(time_seconds % 60)
        milliseconds = int((time_seconds - int(time_seconds)) * 1000)
        return f"{minutes}:{seconds:02d}.{milliseconds:03d}"


    def update_fitness(self, action, info):

        """
        Reward = distance progressed * (discount_factor ^ time_elapsed)
        Encourages fast, efficient progress along the track.
        """
        # 1. Compute progress as fraction of total track
        progress_delta = self.compute_progress()

        # 2. Convert fractional progress into real distance
        delta_distance = progress_delta * self.total_length

        # 3. Discount factor (tune as needed)
        discount_factor = 0.98  # Per second

        # 4. Compute total elapsed time (in seconds)
        time_elapsed = self.time

        # 5. Apply exponential discount
        discounted_reward = delta_distance * (discount_factor ** time_elapsed)

        return discounted_reward

      
    def compute_progress(self):
        """
        Computes the percentage of progress made along the track centerline (0.0 to 1.0).
        Returns:
            progress_delta (float): Progress since last step as fraction of total track length.
        """
        dists = np.linalg.norm(self.path - self.car.pos, axis=1)
        idx = np.argmin(dists)

        current_length = self.cumulative_lengths[idx]
        current_progress = current_length / self.total_length

        delta = current_progress - self.prev_progress

        # Fix wrap-around error when crossing the lap start line
        if delta < -0.5:
            delta += 1.0

        self.prev_progress = current_progress
        return delta


    def check_lap_complete(self):
        """
        Check if the car has crossed the finish line after minimum lap time.
        Returns True if lap is completed, else False.
        """
        if self.time < self.min_lap_time:
            return False  # too early to complete lap

        if len(self.positions) < 2:
            return False  # not enough position history to detect crossing

        prev_pos = Point(self.positions[-2])
        curr_pos = Point(self.positions[-1])
        car_path = LineString([prev_pos, curr_pos])

        # Check if car path crosses the finish line and has not already crossed this lap
        if car_path.crosses(self.finish_line) and not self.crossed_finish_line:
            self.crossed_finish_line = True
            return True
        elif not car_path.crosses(self.finish_line) and self.crossed_finish_line:
            # Reset flag once car moves away from finish line to allow next lap detection
            self.crossed_finish_line = False

        return False

    def render(self, race_line=False):
        '''
        Renders the environment.
        '''
        plot_track_and_trajectory(
            self.track,
            positions=self.positions,
            speeds=self.speeds,
            crash_point=self.crash_point,
            plot_raceline=race_line
        )

    def seed(self, seed=None):
        '''
        Set the random seed for the environment.
        Args:
            seed (int): The random seed to set.
        '''
        np.random.seed(seed)