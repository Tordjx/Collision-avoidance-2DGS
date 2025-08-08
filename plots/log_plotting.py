import numpy as np
import matplotlib.pyplot as plt
import mediapy as media
from tqdm import tqdm

def generate_mock_logs(N=60):
    d = []
    modulated_vel = np.array([0.0, 0.0])
    reference_vel = np.array([0.0, 0.0])
    for _ in range(N):
        current_data = {}

        modulated_vel += np.random.normal(0, 0.1, size=2) * 0.9
        reference_vel += np.random.normal(0, 0.1, size=2) * 0.9

        current_data['modulated_vel'] = modulated_vel.copy()
        current_data['reference_vel'] = reference_vel.copy()
        current_data['correction'] = (modulated_vel - reference_vel).copy()

        current_data['image'] = np.random.normal(0, 1, size=(64, 64))
        current_data["rdot"] = modulated_vel[0]
        current_data["phidot"] = modulated_vel[1]


        d.append(current_data)

    return d

# --- Prepare data ---
logs = generate_mock_logs(60)

# Integrate trajectory
x, y, theta = 0.0, 0.0, 0.0
trajectory = [(x, y)]
dt = 0.1
for entry in logs:
    theta += entry["phidot"] * dt
    x += entry["rdot"] * np.cos(theta) * dt
    y += entry["rdot"] * np.sin(theta) * dt
    trajectory.append((x, y))
trajectory = np.array(trajectory)

# Precompute axis limits for trajectory
margin = 0.2
x_min, x_max = trajectory[:,0].min() - margin, trajectory[:,0].max() + margin
y_min, y_max = trajectory[:,1].min() - margin, trajectory[:,1].max() + margin

# --- Create frames ---
frames = []
for i, entry in tqdm(list(enumerate(logs))):
    fig = plt.figure(figsize=(8, 6))
    # GridSpec: top row split into 2, bottom row spans both
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

    # Camera image
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(entry['image'], cmap='gray')
    ax_img.set_title("Camera Image")
    ax_img.axis('off')

    # Velocity vector widget
    ax_vec = fig.add_subplot(gs[0, 1])
    ax_vec.quiver(0, 0, *entry['reference_vel'], angles='xy', scale_units='xy', scale=1,
                  color='blue', label="Reference")
    ax_vec.quiver(0, 0, *entry['modulated_vel'], angles='xy', scale_units='xy', scale=1,
                  color='orange', label="Modulated")
    ax_vec.quiver(0, 0, *entry['correction'], angles='xy', scale_units='xy', scale=1,
                  color='green', label="Correction")
    ax_vec.set_xlim(-2, 2)
    ax_vec.set_ylim(-2, 2)
    ax_vec.axhline(0, color='k', linewidth=0.5)
    ax_vec.axvline(0, color='k', linewidth=0.5)
    ax_vec.set_aspect('equal', adjustable='box')
    ax_vec.set_title("Velocity Vectors")
    ax_vec.legend(fontsize=6, loc='upper right')

    # Full-width trajectory
    ax_traj = fig.add_subplot(gs[1, :])
    ax_traj.plot(trajectory[:i+1, 0], trajectory[:i+1, 1], 'b-')
    ax_traj.plot(trajectory[0, 0], trajectory[0, 1], 'go', label="Start")
    ax_traj.plot(trajectory[i, 0], trajectory[i, 1], 'ro', label="Current")
    ax_traj.set_xlim(x_min, x_max)
    ax_traj.set_ylim(y_min, y_max)
    ax_traj.set_title("2D Trajectory")
    ax_traj.set_aspect('equal', adjustable='box')
    ax_traj.legend(fontsize=6)
    ax_traj.grid(True)

    plt.tight_layout()

    # Convert to RGB array
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(frame)

    plt.close(fig)

# --- Show video ---
media.show_video(frames, fps=20)
