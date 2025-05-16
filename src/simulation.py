import numpy as np
import shapely.geometry as geom
from shapely.geometry import Point, Polygon
from car import car_agent  # adjust import if needed

def run_simulation(genome, track, spd_init=10.0, dt=0.005, max_time=500):
    """
    Runs a simulation lap using the reactive car_agent and returns telemetry.

    Returns:
    - lap_trace: np.array of positions [x,y] over time
    - crashed: bool, whether car crashed
    - finished: bool, whether lap completed
    - time_sim: total simulation time in seconds
    """

    # Unpack track data
    x_c = track["x_c"]
    y_c = track["y_c"]
    x_l = track["x_l"]
    y_l = track["y_l"]
    x_r = track["x_r"]
    y_r = track["y_r"]
    center_curve = np.stack((x_c, y_c), axis=1)
    hdg_nxt = track["heading"]

    center_line = geom.LineString(center_curve)
    left_line = geom.LineString(np.stack((x_l, y_l), axis=1))
    right_line = geom.LineString(np.stack((x_r, y_r), axis=1))

    # Construct track polygon by joining left and reversed right boundaries
    track_polygon_coords = list(left_line.coords) + list(right_line.coords)[::-1]
    track_polygon = Polygon(track_polygon_coords)

    # Initial car state
    pos_car = np.array([x_c[0], y_c[0]])
    hdg_init = hdg_nxt[0]
    v_car = np.array([spd_init * np.cos(hdg_init), spd_init * np.sin(hdg_init)])

    lap_trace = []
    time_sim = 0.0

    finished = False
    crashed = False

    while not finished and not crashed and time_sim < max_time:
        dis2wall_r = geom.Point(pos_car).distance(right_line)
        dis2wall_l = geom.Point(pos_car).distance(left_line)
        dis2center = geom.Point(pos_car).distance(center_line)

        v_car, nearest_center, idx_closest, spd, hdg = car_agent(
            v_car, pos_car, dis2wall_r, dis2wall_l, dis2center,
            center_curve, hdg_nxt, genome)

        pos_car = pos_car + v_car * dt
        lap_trace.append(pos_car.copy())
        time_sim += dt

        # Precise crash detection: check if car is inside track polygon
        point = Point(pos_car[0], pos_car[1])
        if not track_polygon.contains(point):
            crashed = True
            print(f"Crash detected at {time_sim:.2f}s, position {pos_car}")
            break  # exit simulation early on crash

        # Lap completion condition (back at start line after some time)
        if idx_closest == 0 and time_sim > 10.0:
            finished = True
            print(f"Lap completed in {time_sim:.2f}s")

    return np.array(lap_trace), crashed, finished, time_sim
