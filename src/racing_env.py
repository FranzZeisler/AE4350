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
    
    def __init__(self, track_name, dt, crash_penalty, lap_complete_reward, progress_reward_scale, acceleration_reward, steering_penalty, speed_reward):
        '''
        Initialize the Racing Environment.
        Args:
            track_name (str): The name of the track to load.
            dt (float): Time step for the simulation.
            crash_penalty (float): Penalty for crashing.
            lap_complete_reward (float): Reward for completing a lap.
            progress_reward_scale (float): Scaling factor for progress reward.
            acceleration_reward (float): Reward for acceleration.
            steering_penalty (float): Penalty for steering.
            speed_reward (float): Reward for speed.
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
        self.prev_steering_angle = 0.0
        self.sim_index = 0
        self.discount = 0.98

        # Variables for Rendering
        self.positions = []
        self.speeds = []
        self.crash_point = None

        # Set the parameters for rewards
        self.crash_penalty = crash_penalty
        self.lap_complete_reward = lap_complete_reward
        self.progress_reward_scale = progress_reward_scale
        self.acceleration_reward = acceleration_reward
        self.steering_penalty = steering_penalty
        self.speed_reward = speed_reward

        # Define finish line as a small line segment perpendicular to track start direction
        start_point = self.path[0]
        direction_vector = self.path[1] - self.path[0]
        perp_vector = np.array([-direction_vector[1], direction_vector[0]])
        perp_vector = perp_vector / np.linalg.norm(perp_vector)  # normalize

        line_length = 5.0  # length of finish line segment (meters)
        p1 = start_point + perp_vector * (line_length / 2)
        p2 = start_point - perp_vector * (line_length / 2)
        self.finish_line = LineString([p1, p2])

        self.crossed_finish_line = False  # Flag to detect crossing only once per lap

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
        self.prev_steering_angle = self.car.steering_angle
        self.sim_index = 0

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
        elif self.time >= self.max_time:
            done = True
            info["termination"] = "timeout"

        # Calculate reward
        #reward = self.update_fitness(action, done, info)
        reward = self.alternative_update_fitness(action)

        # Save previous steering angle for smoothness penalty next step
        self.prev_steering_angle = self.car.steering_angle

        return features, reward, done, info

    def format_lap_time(self, time_seconds):
        minutes = int(time_seconds // 60)
        seconds = int(time_seconds % 60)
        milliseconds = int((time_seconds - int(time_seconds)) * 1000)
        return f"{minutes}:{seconds:02d}.{milliseconds:03d}"

    def update_fitness(self, action, done=False, info=None):
        if info is None:
            info = {}

        # Terminal rewards
        if done:
            if info.get("termination") == "crash":
                return self.crash_penalty
            elif info.get("termination") == "lap_complete":
                return self.lap_complete_reward
            
        progress_delta = self.compute_progress()
        progress_bonus = progress_delta * self.progress_reward_scale

        accel_bonus = self.acceleration_reward * action[1]

        steer_penalty = self.steering_penalty * (action[0] ** 2)  # quadratic penalty for steering

        speed_norm = self.car.speed / self.car.max_speed
        speed_bonus = self.speed_reward * speed_norm

        total_reward = progress_bonus + accel_bonus + speed_bonus + steer_penalty

        return total_reward


    def alternative_update_fitness(self, action):
        """
        Alternative reward calculation using discounted immediate progress reward.
        Reward is: delta_distance_travelled * (discount ** (sim_index * dt))
        """
        # Calculate progress delta (distance travelled fraction)
        progress_delta = self.compute_progress()  # fraction of total track length
        delta_distance_travelled = progress_delta * self.total_length  # convert fraction to meters

        # Discounted reward
        discounted_reward = delta_distance_travelled * (self.discount ** (self.sim_index * self.dt))

        # Also add acceleration incentive
        accel_bonus = self.acceleration_reward * action[1]
        
        # Total reward is the sum of discounted progress and speed bonus
        total_reward = discounted_reward + accel_bonus

        return total_reward
    

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
        
        prev_progress = self.prev_progress
        self.prev_progress = current_progress
        
        return current_progress - prev_progress

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
