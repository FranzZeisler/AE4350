import numpy as np
import matplotlib.pyplot as plt
from track_load import load_track
from car import car_agent
import shapely.geometry as geom

# Load track data
track_name = "Austin"
track = load_track(track_name)

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

# Adjust genome to be more cautious (tune these values as you see fit)
genome = [0.4, 0.5, 8, 5, 15, 0.3]  # steer_gain, center_tol, heading_tol, future_short, future_long, accel_gain

dt = 0.005
t_max = 500
steps = int(t_max / dt)

pos_car = np.array([x_c[0], y_c[0]])
hdg_init = hdg_nxt[0]
spd_init = 1.0  # Start very slow
v_car = np.array([spd_init * np.cos(hdg_init), spd_init * np.sin(hdg_init)])

lap_trace = []
speed_trace = []
time_trace = []

for step in range(steps):
    time_sim = step * dt

    dis2wall_r = geom.Point(pos_car).distance(right_line)
    dis2wall_l = geom.Point(pos_car).distance(left_line)
    dis2center = geom.Point(pos_car).distance(center_line)

    # Simple crash condition: if too far from centerline or outside boundaries
    track_half_width = min(dis2wall_r + dis2center, dis2wall_l + dis2center)
    if dis2center > track_half_width - 0.1:
        print(f"Crash at time {time_sim:.2f}s, position {pos_car}")
        break

    # Run agent control
    v_car, nearest_center, idx_closest, spd, hdg = car_agent(
        v_car, pos_car, dis2wall_r, dis2wall_l, dis2center,
        center_curve, hdg_nxt, genome
    )

    # Slow down if drifting too far from centerline
    if dis2center > genome[1] * 1.5 and spd > 5:
        spd *= 0.8
        new_heading_rad = np.radians(hdg)
        v_car = np.array([spd * np.cos(new_heading_rad), spd * np.sin(new_heading_rad)])

    pos_car = pos_car + v_car * dt

    lap_trace.append(pos_car.copy())
    speed_trace.append(spd)
    time_trace.append(time_sim)

    # Check for lap completion
    if idx_closest == 0 and time_sim > 5:
        print(f"Lap completed at {time_sim:.2f} s")
        break

lap_trace = np.array(lap_trace)

# Plotting
plt.figure(figsize=(10, 8))
plt.plot(x_c, y_c, color="yellow", label="Centerline")
plt.plot(x_l, y_l, color="blue", label="Left Boundary")
plt.plot(x_r, y_r, color="blue", label="Right Boundary")
plt.plot(lap_trace[:, 0], lap_trace[:, 1], color="red", linewidth=2, label="Agent Path")
plt.axis("equal")
plt.title(f"Reactive F1 Agent Lap - {track_name.upper()}")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
