import matplotlib.pyplot as plt 
from track import load_track

track_name = "Austin"
# Load and unpack
track = load_track(track_name)

x_c, y_c = track["x_c"], track["y_c"]
x_l, y_l = track["x_l"], track["y_l"]
x_r, y_r = track["x_r"], track["y_r"]
raceline = track["raceline"]

# === Plot ===
plt.figure(figsize=(10, 8))
plt.plot(x_c, y_c, label="Centerline", color="orange")
plt.plot(x_l, y_l, label="Left Boundary", color="blue")
plt.plot(x_r, y_r, label="Right Boundary", color="teal")

if raceline is not None:
    plt.plot(raceline[:, 0], raceline[:, 1], label="Reference Raceline", linestyle="--", color="black")

plt.axis("equal")
plt.title(f"Track: {track_name.capitalize()}")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
