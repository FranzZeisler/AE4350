import matplotlib.pyplot as plt 
from track import load_track

track_name = "Austin"  # Example track name
# Load and unpack
track = load_track(track_name)

# Unpack the track data
x_c, y_c = track["x_c"], track["y_c"]
x_l, y_l = track["x_l"], track["y_l"]
x_r, y_r = track["x_r"], track["y_r"]
raceline = track["raceline"]

# === Plot ===
plt.figure(figsize=(10, 8))
plt.plot(x_c, y_c, label="Centerline", color="orange")
# Plot left and right boundaries, but only label one to avoid duplicate legend entries
plt.plot(x_l, y_l, color="blue", label="Boundary")
plt.plot(x_r, y_r, color="blue")
plt.plot(raceline[:, 0], raceline[:, 1], label="Reference Raceline", linestyle="--", color="black")

# Plot the track boundaries
plt.axis("equal")
plt.title(f"Track: {track_name}")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
