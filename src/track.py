import os
import numpy as np
import pandas as pd
from shapely import LinearRing, Point, Polygon

class Track:
    def __init__(self, track_name, base_path="data"):
        """
        Initialize the track object by loading the necessary track data.
        :param track_name: Name of the track (without .csv).
        :param base_path: Base path to the 'data' directory.
        """
        self.track_name = track_name
        self.base_path = base_path
        
        # Load the track data
        self.load_track()

    def load_track(self):
        """
        Load and process the racetrack from the TUMFTM dataset.
        """
        track_path = os.path.join(self.base_path, "tracks", f"{self.track_name}.csv")

        if not os.path.exists(track_path):
            raise FileNotFoundError(f"Track file not found: {track_path}")

        # Load track centerline and widths
        df = pd.read_csv(track_path, comment='#', header=None,
                         names=["x_m", "y_m", "w_tr_right_m", "w_tr_left_m"])

        self.x_c = df["x_m"].to_numpy()
        self.y_c = df["y_m"].to_numpy()
        self.w_r = df["w_tr_right_m"].to_numpy()
        self.w_l = df["w_tr_left_m"].to_numpy()

        # Compute heading between consecutive centerline points
        dx = np.roll(self.x_c, -1) - self.x_c
        dy = np.roll(self.y_c, -1) - self.y_c
        self.heading = np.arctan2(dy, dx)

        # Fix discontinuity in headings (ensure continuous wrapping)
        heading_diff = np.diff(self.heading)
        heading_diff[heading_diff > np.pi] -= 2 * np.pi  # Adjust for angle overflow
        heading_diff[heading_diff < -np.pi] += 2 * np.pi  # Adjust for angle underflow
        self.heading[1:] = np.cumsum(heading_diff)  # Accumulate the adjusted differences

        # Calculate boundary coordinates by shifting centerline points perpendicular to heading
        heading_left = self.heading + np.pi / 2
        heading_right = self.heading - np.pi / 2
        self.x_l = self.x_c + self.w_l * np.cos(heading_left)
        self.y_l = self.y_c + self.w_l * np.sin(heading_left)
        self.x_r = self.x_c + self.w_r * np.cos(heading_right)
        self.y_r = self.y_c + self.w_r * np.sin(heading_right)


    def project(self, x, y):
        """
        Project the car's position onto the track's centerline to determine progress.
        :param x: Car's x position
        :param y: Car's y position
        :return: Normalized progress along the centerline (0 to 1)
        """
        # Find the closest point on the track centerline to (x, y)
        min_distance = np.inf
        closest_point = -1
        for i in range(len(self.x_c)):
            dist = np.sqrt((self.x_c[i] - x) ** 2 + (self.y_c[i] - y) ** 2)
            if dist < min_distance:
                min_distance = dist
                closest_point = i

        # Calculate the total length of the track
        total_length = 0
        for i in range(1, len(self.x_c)):
            segment_length = np.sqrt((self.x_c[i] - self.x_c[i-1])**2 + (self.y_c[i] - self.y_c[i-1])**2)
            total_length += segment_length

        # Calculate the progress along the track as the distance from the start to the closest point
        progress = 0
        for i in range(1, closest_point + 1):
            segment_length = np.sqrt((self.x_c[i] - self.x_c[i-1])**2 + (self.y_c[i] - self.y_c[i-1])**2)
            progress += segment_length

        # Normalize the progress to [0, 1]
        normalized_progress = progress / total_length if total_length > 0 else 0.0
        return normalized_progress

    def build_track_polygon(self):
        """
        Build a closed polygon representing the track boundaries.

        Returns:
        - shapely.geometry.Polygon object representing the drivable area of the track
        """
        # Left and right boundary points
        left_boundary = np.column_stack((self.x_l, self.y_l))
        right_boundary = np.column_stack((self.x_r, self.y_r))  # Reverse right boundary to close polygon properly

        # Create linear rings to ensure boundaries are closed loops
        left_ring = LinearRing(left_boundary)
        right_ring = LinearRing(right_boundary[::-1])  # Reverse to close the right boundary properly

        # Combine boundary coordinates to form polygon
        polygon_points = np.vstack((left_ring.coords, right_ring.coords))

        # Create a Polygon from the coordinates
        poly = Polygon(polygon_points)

        # Fix invalid polygons by buffering with zero-width (often fixes self-intersections)
        if not poly.is_valid:
            poly = poly.buffer(0)
            if not poly.is_valid:
                raise ValueError("Track polygon is invalid and could not be fixed.")

        return poly

    def is_off_track(self, x, y):
        """
        Check if the car is off track using the track's boundary polygon.

        :param x: Car's x position
        :param y: Car's y position
        :return: True if off track, else False
        """
        # Build the track polygon using the provided function
        track_polygon = self.build_track_polygon()
        
        # Check if the car's position is inside the track polygon
        car_position = Point(x, y)
        
        # Return True if the car is outside the track (off-track), else False
        return not track_polygon.contains(car_position)


    def compute_errors(self, x, y, theta):
        """
        Compute the lateral and heading errors.
        :param x: Car's x position
        :param y: Car's y position
        :param theta: Car's heading
        :return: Lateral error and heading error
        """
        # Compute lateral error as the perpendicular distance from the car to the centerline
        progress = self.project(x, y)
        track_direction = self.heading[progress]
        lateral_error = (x - self.x_c[progress]) * np.sin(track_direction) - (y - self.y_c[progress]) * np.cos(track_direction)
        heading_error = theta - track_direction  # Heading error
        return lateral_error, heading_error
