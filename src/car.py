import numpy as np

class Car:
    def __init__(self, x, y, heading):
        self.pos = np.array([x, y], dtype=float)
        self.heading = heading  # radians
        self.speed = 10.0        # m/s
        self.velocity = np.array([0.0, 0.0])
        self.wheelbase = 3.6    # meters

        self.max_accel = 15.0       # m/s²
        self.max_decel = 30.0       # m/s²
        self.max_lateral_g = 4.0 * 9.81
        self.top_speed = 100.0      # m/s
        self.max_steer_rate = np.deg2rad(2.0)  # radians per step
        self.steering_angle = 0.0   # radians
        self.dt = 0.1            # seconds

        self.last_lat_accel = 0.0
        self.last_centripetal = 0.0
        self.last_vel_heading = 0.0

    def update(self, target_steer, throttle):
        steer_diff = np.clip(target_steer - self.steering_angle, -self.max_steer_rate, self.max_steer_rate)
        self.steering_angle += steer_diff

        if throttle >= 0:
            accel = throttle * self.max_accel
        else:
            accel = throttle * self.max_decel

        self.speed += accel * self.dt
        self.speed = np.clip(self.speed, 0, self.top_speed)

        turning_radius = self.wheelbase / (np.tan(self.steering_angle) + 1e-6)
        lat_accel = self.speed ** 2 / turning_radius
        if abs(lat_accel) > self.max_lateral_g:
            self.speed = np.sqrt(abs(self.max_lateral_g * turning_radius))

        angular_velocity = self.speed / self.wheelbase * np.tan(self.steering_angle)
        self.heading += angular_velocity * self.dt

        dx = self.speed * np.cos(self.heading) * self.dt
        dy = self.speed * np.sin(self.heading) * self.dt
        self.velocity = np.array([dx, dy]) / self.dt
        self.pos += np.array([dx, dy])

        self.last_lat_accel = lat_accel
        self.last_centripetal = lat_accel
        self.last_vel_heading = self.velocity_heading()

    def velocity_heading(self):
        vel_angle = np.arctan2(self.velocity[1], self.velocity[0])
        angle_diff = vel_angle - self.heading
        return (angle_diff + np.pi) % (2 * np.pi) - np.pi

    def _heading_error(self, target_point):
        vec_to_target = target_point - self.pos
        target_heading = np.arctan2(vec_to_target[1], vec_to_target[0])
        angle_error = target_heading - self.heading
        return (angle_error + np.pi) % (2 * np.pi) - np.pi

    def _estimate_curvatures(self, path_points, start_idx, count=5):
        curvs = []
        for i in range(count):
            i0 = (start_idx + i) % len(path_points)
            i1 = (i0 + 1) % len(path_points)
            i2 = (i1 + 1) % len(path_points)

            p0, p1, p2 = path_points[i0], path_points[i1], path_points[i2]
            a = np.linalg.norm(p0 - p1)
            b = np.linalg.norm(p1 - p2)
            c = np.linalg.norm(p2 - p0)
            s = (a + b + c) / 2
            area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))
            if a * b * c != 0:
                curvature = 4 * area / (a * b * c)
            else:
                curvature = 0
            curvs.append(curvature)
        return curvs

    def _raycast_to_boundary(self, track, angle):
        # Simple ray to boundary (approximate by closest point)
        ray_dir = np.array([np.cos(angle), np.sin(angle)])
        distances = []

        for side in ['x_l', 'x_r']:
            side_x = track[side]
            side_y = track['y_' + side[-1]]
            edge_points = np.stack((side_x, side_y), axis=1)
            dists = np.linalg.norm(edge_points - self.pos, axis=1)
            min_dist = np.min(dists)
            distances.append(min_dist)

        return min(distances)

    def get_state(self, track, path_points):
        state = []
        lidar_angles = np.deg2rad([-90, -45, -15, 0, 15, 45, 90])
        for angle in lidar_angles:
            lidar_dir = self.heading + angle
            distance = self._raycast_to_boundary(track, lidar_dir)
            state.append(distance)

        path_deltas = path_points - self.pos
        dists = np.linalg.norm(path_deltas, axis=1)
        closest_idx = np.argmin(dists)
        current_heading_error = self._heading_error(path_points[closest_idx])
        future_idx = (closest_idx + 20) % len(path_points)
        future_heading_error = self._heading_error(path_points[future_idx])
        state.append(current_heading_error)
        state.append(future_heading_error)

        curvatures = self._estimate_curvatures(path_points, closest_idx, count=5)
        state.extend(curvatures)

        state.append(self.speed)
        state.append(self.last_lat_accel)
        state.append(self.last_centripetal)
        state.append(self.last_vel_heading)

        return np.array(state, dtype=float)
