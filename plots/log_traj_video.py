import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re
import os
import numpy as np
import pinocchio as pin
def integrate(x,y,phi, rdot, phidot):
    se2 = pin.liegroups.SE2()
    dt = 1e-1
    pose = np.array([x,y, np.cos(phi), np.sin(phi)])
    twist = np.array([rdot, 0 , phidot])
    new_pose = se2.integrate(pose, twist * dt)
    x = new_pose[0]
    y = new_pose[1]
    phi = np.arctan2(new_pose[3], new_pose[2])
    return x, y , phi
log_folder = "/home/vtordjma/Documents/log_expe_icra"
output_folder = os.path.join(log_folder, "output_videos")
os.makedirs(output_folder, exist_ok=True)

def parse_experiment(filepath):
    joystick, actions, rdot, phidot = [], [], [], []
    with open(filepath, "r") as f:
        for line in f:
            match_action = re.search(r"Action=\[(.*?)\], Joystick", line)
            match_joystick = re.search(r"Joystick=\[(.*?)\], rdot", line)
            match_rdot = re.search(r"rdot=([-\d.eE]+)", line)
            match_phidot = re.search(r"phidot=([-\d.eE]+)", line)

            if match_action and match_joystick and match_rdot and match_phidot:
                action = [float(v) for v in match_action.group(1).split()]
                joy = [float(v.replace(",", "")) for v in match_joystick.group(1).split()]
                actions.append(action)
                joystick.append(joy)
                rdot.append(float(match_rdot.group(1)))
                phidot.append(float(match_phidot.group(1)))

    return np.array(actions), np.array(joystick), np.array(rdot), np.array(phidot)

def split_by_joystick(actions, joysticks, rdot, phidot, target=[0.001, -1.0], tol=1e-6):
    segments = []
    start_idx = None
    def matches(j): return np.allclose(j, target, atol=tol)

    for i, joy in enumerate(joysticks):
        if matches(joy):
            if start_idx is None: start_idx = i
        else:
            if start_idx is not None:
                segments.append({
                    "actions": actions[start_idx:i],
                    "joysticks": joysticks[start_idx:i],
                    "rdot": rdot[start_idx:i],
                    "phidot": phidot[start_idx:i]
                })
                start_idx = None
    if start_idx is not None:
        segments.append({
            "actions": actions[start_idx:],
            "joysticks": joysticks[start_idx:],
            "rdot": rdot[start_idx:],
            "phidot": phidot[start_idx:]
        })
    return [s for s in segments if len(s['actions']) > 20]

def integrate_segment(segment, dt=0.1):
    x, y, phi = 0, 0, 0
    X, Y, commanded_vels, modulated_vels = [], [], [], []
    actions, joysticks, rdots, phidots = segment['actions'], segment['joysticks'], segment['rdot'], segment['phidot']

    for i in range(len(rdots)):
        X.append(x)
        Y.append(y)

        commanded_vel = np.array([-joysticks[i][0], -joysticks[i][1]])
        modulated_vel = np.array([-actions[i][1], -actions[i][0]])
        commanded_vels.append(commanded_vel)
        modulated_vels.append(modulated_vel)

        x,y,phi = integrate(x,y,phi, rdots[i],phidots[i])

    X = np.array(X)
    Y = np.array(Y)
    commanded_vels = np.array(commanded_vels)
    modulated_vels = np.array(modulated_vels)
    return X, Y, commanded_vels, modulated_vels

def create_experiment_video(segments, video_path, dt=0.1):
    fig, (ax_traj, ax_vel) = plt.subplots(1, 2, figsize=(12, 6))

    # Precompute axis limits per segment
    segment_limits = []
    integrated_segments = []
    for seg in segments:
        X, Y, commanded, modulated = integrate_segment(seg, dt)
        integrated_segments.append((X, Y, commanded, modulated))
        xlim = (X.min() - 0.1, X.max() + 0.1)
        ylim = (Y.min() - 0.1, Y.max() + 0.1)
        segment_limits.append((xlim, ylim))

    # Concatenate all segments for animation
    frames = []
    segment_start_frames = [0]
    for X, _, _, _ in integrated_segments:
        segment_start_frames.append(segment_start_frames[-1] + len(X))

    total_frames = segment_start_frames[-1]

    def update(frame):
        # Determine which segment we're in
        seg_idx = max(i for i, start in enumerate(segment_start_frames) if frame >= start)
        local_frame = frame - segment_start_frames[seg_idx]

        X, Y, commanded, modulated = integrated_segments[seg_idx]
        xlim, ylim = segment_limits[seg_idx]

        # Trajectory subplot
        ax_traj.clear()
        ax_traj.set_xlim(xlim)
        ax_traj.set_ylim(ylim)
        ax_traj.set_xlabel("X position")
        ax_traj.set_ylabel("Y position")
        ax_traj.grid(True)
        ax_traj.plot(X[:local_frame], Y[:local_frame], color='black', alpha=0.7)

        # Velocity quadrant subplot
        ax_vel.clear()
        ax_vel.set_xlim(-1.0, 1.0)
        ax_vel.set_ylim(-1.0, 1.0)
        ax_vel.set_xlabel("Velocity X")
        ax_vel.set_ylabel("Velocity Y")
        ax_vel.set_title("Velocities")
        ax_vel.grid(True)
        ax_vel.plot([0,0],[ax_vel.get_ylim()[0],ax_vel.get_ylim()[1]], 'k--', alpha=0.5)
        ax_vel.plot([ax_vel.get_xlim()[0],ax_vel.get_xlim()[1]],[0,0], 'k--', alpha=0.5)
        ax_vel.quiver(0, 0, commanded[local_frame,0], commanded[local_frame,1],
                      color='blue', scale=1, scale_units='xy', angles='xy', width=0.01)
        ax_vel.quiver(0, 0, modulated[local_frame,0], modulated[local_frame,1],
                      color='red', scale=1, scale_units='xy', angles='xy', width=0.01)

    ani = animation.FuncAnimation(fig, update, frames=total_frames, repeat=False)
    writer = animation.FFMpegWriter(fps=int(1/dt))
    ani.save(video_path, writer=writer)
    plt.close(fig)

# --- main loop ---
folders = [f for f in os.listdir(log_folder) if "CAM" not in f and "output" not in f]

for folder in folders:
    files = os.listdir(os.path.join(log_folder, folder))
    for file in files:
        experiment_file = os.path.join(log_folder, folder, file)
        actions, joysticks, rdot, phidot = parse_experiment(experiment_file)
        segments = split_by_joystick(actions, joysticks, rdot, phidot)
        print(f"Found {len(segments)} segments in {file}")

        method = "foa" if "foa" in file else "rl"
        timestamp = os.path.splitext(file)[0]

        video_path = os.path.join(output_folder, f"{folder}_{method}_{timestamp}.mp4")
        create_experiment_video(segments, video_path)
        print(f"Saved video: {video_path}")

print("All experiment videos created!")
