import numpy as np

class PIDController:
    def __init__(self, kp=2.0, ki=0.1, kd=0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def control(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

def pid_steering_control(car, path_points, lookahead=5):
    """Steering PID based on heading error to a lookahead point."""
    closest_idx = np.argmin(np.linalg.norm(path_points - car.pos, axis=1))
    target_idx = min(closest_idx + lookahead, len(path_points) - 1)
    target_point = path_points[target_idx]
    
    dx = target_point[0] - car.pos[0]
    dy = target_point[1] - car.pos[1]
    target_heading = np.arctan2(dy, dx)

    heading_error = (target_heading - car.heading + np.pi) % (2 * np.pi) - np.pi
    return heading_error
