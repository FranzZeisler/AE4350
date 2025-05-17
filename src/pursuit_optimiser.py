from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import numpy as np

from simulate_pursuit import simulate_track_pursuit
from track import load_track

# ──────────────────────────────────────────────────────────────
# 1. Track lists
# ──────────────────────────────────────────────────────────────
training_tracks = [
    "Austin", "Budapest", "Catalunya", "Hockenheim", "Melbourne", 
    "MexicoCity", "Montreal", "Monza", "Sakhir", "SaoPaulo",
    "Sepang", "Shanghai", "Silverstone", "Sochi"
]

testing_tracks = ["Spa", "Spielberg", "Suzuka", "YasMarina", "Zandvoort"]

# ──────────────────────────────────────────────────────────────
# 2. Optimization space
# ──────────────────────────────────────────────────────────────
space = [
    Real(1.0, 10.0, name='base_lookahead'),
    Real(0.05, 1.0, name='lookahead_gain'),
    Real(0.1, 1.0, name='alpha'),
    Integer(1, 15, name='throttle_threshold_1'),
    Integer(10, 40, name='throttle_threshold_2'),
    Real(0.7, 1.0, name='throttle_1'),     
    Real(0.3, 0.8, name='throttle_2'),
    Real(0.1, 0.6, name='throttle_3'),
]

# ──────────────────────────────────────────────────────────────
# 3. Objective: penalize crashes, reward good avg times
# ──────────────────────────────────────────────────────────────
@use_named_args(space)
def objective(**params):
    lap_times = []
    invalid_tracks = 0
    valid_tracks = 0
    total_time = 0.0

    for track in training_tracks:
        training_track = load_track(track)
        lap_time = simulate_track_pursuit(
            training_track,
            base_lookahead       = params['base_lookahead'],
            lookahead_gain       = params['lookahead_gain'],
            alpha                = params['alpha'],
            throttle_threshold_1 = params['throttle_threshold_1'],
            throttle_threshold_2 = params['throttle_threshold_2'],
            throttle_1           = params['throttle_1'],
            throttle_2           = params['throttle_2'],
            throttle_3           = params['throttle_3'],
            plot_speed=False
        )
        if lap_time == 999.0:
            invalid_tracks += 1
            if invalid_tracks > 5:
                break
        else:
            lap_times.append(lap_time)
            valid_tracks += 1
            total_time += lap_time

        
    if valid_tracks == 0:
        avg_time = 999.0
    else:
        avg_time = total_time / valid_tracks

    penalty = invalid_tracks * 100
    print(f"✓ Valid: {valid_tracks}/14 | Avg: {avg_time:.2f}s | Penalty: {penalty:.0f}")
    return avg_time + penalty

# ──────────────────────────────────────────────────────────────
# 4. Run optimization
# ──────────────────────────────────────────────────────────────
result = gp_minimize(
    objective,
    space,
    n_calls=50,
    n_initial_points=30,
    random_state=42,
    verbose=True
)

# ──────────────────────────────────────────────────────────────
# 5. Best result summary
# ──────────────────────────────────────────────────────────────
print("\n╭─ Best average lap time: {:.2f}s ───────────────────────────╮".format(result.fun))
for name, val in zip([dim.name for dim in space], result.x):
    print(f"│ {name:<20}= {val:>8.4f}")
print("╰─────────────────────────────────────────────────────────────╯")

# ──────────────────────────────────────────────────────────────
# 6. Visualize results on all training tracks
# ──────────────────────────────────────────────────────────────
print("\nEvaluating best params on all training tracks:")
for track in training_tracks:
    t = load_track(track)
    lap_time = simulate_track_pursuit(
        t,
        base_lookahead       = result.x[0],
        lookahead_gain       = result.x[1],
        alpha                = result.x[2],
        throttle_threshold_1 = result.x[3],
        throttle_threshold_2 = result.x[4],
        throttle_1           = result.x[5],
        throttle_2           = result.x[6],
        throttle_3           = result.x[7],
        plot_speed=True
    )
    print(f"Track {track}: Lap time = {lap_time:.2f}s")

# ──────────────────────────────────────────────────────────────
# 7. Test on unseen tracks
# ──────────────────────────────────────────────────────────────
print("\nEvaluating on unseen test tracks:")
for track in testing_tracks:
    t = load_track(track)
    lap_time = simulate_track_pursuit(
        t,
        base_lookahead       = result.x[0],
        lookahead_gain       = result.x[1],
        alpha                = result.x[2],
        throttle_threshold_1 = result.x[3],
        throttle_threshold_2 = result.x[4],
        throttle_1           = result.x[5],
        throttle_2           = result.x[6],
        throttle_3           = result.x[7],
        plot_speed=True
    )
    print(f"Track {track}: Lap time = {lap_time:.2f}s")
