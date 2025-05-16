import numpy as np
import matplotlib.pyplot as plt
from track import load_track
from nn_agent import car_agent, car_agent_forward_only
from shapely.geometry import LineString, Point, Polygon

# —————————————————————————————————————————————————————————————————————————
# Helper to bias NN throttle so car actually moves
# —————————————————————————————————————————————————————————————————————————
def make_biased_genome(throttle_bias=2.0):
    """
    Create a genome of length GENOME_LENGTH with:
      - all weights zero
      - the output-layer throttle bias set to `throttle_bias`
    That pushes tanh(bias) -> ~1, so full acceleration.
    """
    # Must match the same GENOME_LENGTH in your agent module:
    from nn_agent import GENOME_LENGTH, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
    g = np.zeros(GENOME_LENGTH, dtype=float)

    # index of the second output bias b2:
    #   i2h = INPUT_SIZE*HIDDEN_SIZE
    #   hb  = HIDDEN_SIZE
    #   h2o = HIDDEN_SIZE*OUTPUT_SIZE
    #   then biases start at i2h+hb+h2o
    bias_start = INPUT_SIZE*HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE*OUTPUT_SIZE
    g[bias_start + 1] = throttle_bias
    return g

# —————————————————————————————————————————————————————————————————————————
# Load track
# —————————————————————————————————————————————————————————————————————————
track_name = "Austin"
track = load_track(track_name)

x_c, y_c = track["x_c"], track["y_c"]
x_l, y_l = track["x_l"], track["y_l"]
x_r, y_r = track["x_r"], track["y_r"]

center_curve = np.stack((x_c, y_c), axis=1)
hdg_nxt      = track["heading"]

left_line   = LineString(np.stack((x_l, y_l), axis=1))
right_line  = LineString(np.stack((x_r, y_r), axis=1))

# build polygon for crash detection
poly_coords = list(left_line.coords) + list(right_line.coords)[::-1]
track_polygon = Polygon(poly_coords)

# —————————————————————————————————————————————————————————————————————————
# Simulation parameters
# —————————————————————————————————————————————————————————————————————————
dt     = 0.005
t_max  = 500.0
steps  = int(t_max / dt)
spd_init = 1.0

# initial state
pos_car = np.array([x_c[0], y_c[0]])
v_car   = np.array([spd_init * np.cos(hdg_nxt[0]),
                    spd_init * np.sin(hdg_nxt[0])])

# Choose your agent:
USE_FORWARD_ONLY = False

if USE_FORWARD_ONLY:
    agent = car_agent_forward_only
    genome = None
else:
    agent = car_agent
    genome = make_biased_genome(throttle_bias=2.0)  # strong forward thrust

# telemetry storage
lap_trace   = []
speed_trace = []
time_trace  = []

# for NN agent state
prev_state = None

# —————————————————————————————————————————————————————————————————————————
# Run simulation
# —————————————————————————————————————————————————————————————————————————
for step in range(steps):
    t = step*dt

    if USE_FORWARD_ONLY:
        # forward-only agent: signature is same except no genome
        v_car, nearest_pt, idx, spd, hdg_deg, prev_state = agent(
            v_car, pos_car,
            center_curve, hdg_nxt,
            left_line, right_line,
            prev_state=None
        )
    else:
        # NN agent
        v_car, nearest_pt, idx, spd, hdg_deg, prev_state = agent(
            v_car, pos_car,
            center_curve, hdg_nxt,
            left_line, right_line,
            genome,
            prev_state=prev_state
        )

    # advance
    pos_car = pos_car + v_car * dt

    lap_trace.append(pos_car.copy())
    speed_trace.append(spd)
    time_trace.append(t)

    # crash?
    if not track_polygon.contains(Point(pos_car)):
        print(f"CRASH at t={t:.2f}s, pos={pos_car}")
        break

    # lap done?
    if idx == 0 and t > 5.0:
        print(f"LAP FINISHED in {t:.2f}s")
        break

lap_trace = np.array(lap_trace)

# —————————————————————————————————————————————————————————————————————————
# Plot
# —————————————————————————————————————————————————————————————————————————
plt.figure(figsize=(10,6))
plt.plot(x_c, y_c, color="yellow", label="Centerline")
plt.plot(x_l, y_l, color="blue",   label="Left boundary")
plt.plot(x_r, y_r, color="blue",   label="Right boundary")
plt.plot(lap_trace[:,0], lap_trace[:,1],
         color="red", linewidth=2, label="Car path")
plt.axis("equal")
plt.title("Agent Path – " + track_name)
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
