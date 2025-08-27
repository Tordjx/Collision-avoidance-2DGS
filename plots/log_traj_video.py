import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re
import os
import numpy as np
import pinocchio as pin

def integrate(x, y, phi, rdot, phidot):
    se2 = pin.liegroups.SE2()
    dt = 0.1
    pose = np.array([x, y, np.cos(phi), np.sin(phi)])
    twist = np.array([rdot, 0, phidot])
    new_pose = se2.integrate(pose, twist * dt)
    x = new_pose[0]
    y = new_pose[1]
    phi = np.arctan2(new_pose[3], new_pose[2])
    return x, y, phi

log_folder = "/home/vtordjma/Documents/log_expe_icra"
output_folder = os.path.join(log_folder, "output_videos")
os.makedirs(output_folder, exist_ok=True)

def parse_experiment(filepath, method="rl"):
    joystick, actions, rdot, phidot, obstacles = [], [], [], [], []
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

                if method == "foa":
                    if "ObstaclePoints=[[" in line:
                        obs_lines = [line.split("ObstaclePoints=[[")[-1]]
                        while "]]" not in obs_lines[-1]:
                            obs_lines.append(next(f).strip())
                        obs_text = "\n".join(obs_lines).replace("]]","").strip()
                        pts_list = []
                        for l in obs_text.split("\n"):
                            l = l.strip().replace("[","").replace("]","")
                            if l:
                                pts_list.append([float(v) for v in l.split()])
                        obstacles.append(np.array(pts_list))
                    else:
                        obstacles.append(np.empty((0,2)))
    return np.array(actions), np.array(joystick), np.array(rdot), np.array(phidot), obstacles

def split_by_joystick(actions, joysticks, rdot, phidot, obstacles, target=[0.001, -1.0], tol=1e-6):
    segments, start_idx = [], None
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
                    "phidot": phidot[start_idx:i],
                    "obstacles": obstacles[start_idx:i]
                })
                start_idx = None
    if start_idx is not None:
        segments.append({
            "actions": actions[start_idx:],
            "joysticks": joysticks[start_idx:],
            "rdot": rdot[start_idx:],
            "phidot": phidot[start_idx:],
            "obstacles": obstacles[start_idx:]
        })
    return segments#[s for s in segments if len(s['actions']) > 20]

def integrate_segment(segment, dt=0.1):
    x, y, phi = 0, 0, 0
    X, Y, commanded_vels, modulated_vels = [], [], [], []
    obstacle_world, accumulated_obstacles = [], []  # NEW: cumulative obstacles

    actions, joysticks, rdots, phidots, obstacles = segment['actions'], segment['joysticks'], segment['rdot'], segment['phidot'], segment['obstacles']

    for i in range(len(rdots)):
        X.append(x)
        Y.append(y)

        commanded_vel = np.array([-joysticks[i][0], -joysticks[i][1]])
        modulated_vel = np.array([-actions[i][1], -actions[i][0]])
        commanded_vels.append(commanded_vel)
        modulated_vels.append(modulated_vel)

        # transform obstacle points to world frame
        if  len(obstacles)>0  and obstacles[i].size > 0:
            c, s = np.cos(phi), np.sin(phi)
            R = np.array([[c, -s],[s, c]])
            obs_world = (R @ obstacles[i].T).T + np.array([x,y])
        else:
            obs_world = np.empty((0,2))

        obstacle_world.append(obs_world)

        # accumulate all obstacles
        if i == 0:
            accumulated_obstacles.append(obs_world)
        else:
            if obs_world.size > 0:
                accumulated_obstacles.append(np.vstack([accumulated_obstacles[-1], obs_world]))
            else:
                accumulated_obstacles.append(accumulated_obstacles[-1])

        x, y, phi = integrate(x, y, phi, rdots[i], phidots[i])

    return np.array(X), np.array(Y), np.array(commanded_vels), np.array(modulated_vels), obstacle_world, accumulated_obstacles

def create_experiment_video(segments, video_path, dt=0.1, method="rl"):
    fig, (ax_traj, ax_vel) = plt.subplots(1, 2, figsize=(12,6))
    integrated_segments, segment_limits = [], []

    for seg in segments:
        X, Y, commanded, modulated, obstacles_world, accumulated_obstacles = integrate_segment(seg, dt)
        integrated_segments.append((X, Y, commanded, modulated, obstacles_world, accumulated_obstacles))

        all_x = np.concatenate([X] + [obs[:,0] for obs in accumulated_obstacles if len(obs)>0])
        all_y = np.concatenate([Y] + [obs[:,1] for obs in accumulated_obstacles if len(obs)>0])
        segment_limits.append((all_x.min()-0.1, all_x.max()+0.1, all_y.min()-0.1, all_y.max()+0.1))

    segment_start_frames = [0]
    for X, *_ in integrated_segments:
        segment_start_frames.append(segment_start_frames[-1] + len(X))
    total_frames = segment_start_frames[-1]

    def update(frame):
        seg_idx = max(i for i,start in enumerate(segment_start_frames) if frame >= start)
        local_frame = frame - segment_start_frames[seg_idx]

        X, Y, commanded, modulated, obstacles_world, accumulated_obstacles = integrated_segments[seg_idx]
        xlim, xmax, ylim, ymax = segment_limits[seg_idx]

        ax_traj.clear()
        ax_traj.set_xlim(xlim, xmax)
        ax_traj.set_ylim(ylim, ymax)
        ax_traj.set_xlabel("X")
        ax_traj.set_ylabel("Y")
        ax_traj.grid(True)
        ax_traj.plot(X[:local_frame], Y[:local_frame], color='black', alpha=0.7)

        # plot accumulated obstacles with fading
        if method=="foa" and len(accumulated_obstacles[local_frame])>0:
            obs = accumulated_obstacles[local_frame]
            alpha_vals = np.linspace(0.1, 0.8, len(obs))
            ax_traj.scatter(obs[:,0], obs[:,1], color='orange', s=30, alpha=alpha_vals)

        ax_vel.clear()
        ax_vel.set_xlim(-2.0, 2.0)
        ax_vel.set_ylim(-2.0, 2.0)
        ax_vel.set_xlabel("Vel X")
        ax_vel.set_ylabel("Vel Y")
        ax_vel.set_title("Velocities")
        ax_vel.grid(True)
        ax_vel.plot([0,0],[ax_vel.get_ylim()[0],ax_vel.get_ylim()[1]],'k--',alpha=0.5)
        ax_vel.plot([ax_vel.get_xlim()[0],ax_vel.get_xlim()[1]],[0,0],'k--',alpha=0.5)
        ax_vel.quiver(0,0,commanded[local_frame,0],commanded[local_frame,1],
                      color='blue', scale=1, scale_units='xy', angles='xy', width=0.01)
        ax_vel.quiver(0,0,modulated[local_frame,0],modulated[local_frame,1],
                      color='red', scale=1, scale_units='xy', angles='xy', width=0.01)

    ani = animation.FuncAnimation(fig, update, frames=total_frames, repeat=False)
    writer = animation.FFMpegWriter(fps=int(1/dt))
    ani.save(video_path, writer=writer)
    plt.close(fig)

# --- main loop ---
folders = [f for f in os.listdir(log_folder) if "CAM" not in f and "output" not in f and "resultat" not in f]
for folder in folders:
    files = os.listdir(os.path.join(log_folder, folder))
    for file in files:
        experiment_file = os.path.join(log_folder, folder, file)
        method = "foa" if "foa" in file else "rl"
        actions, joysticks, rdot, phidot, obstacles = parse_experiment(experiment_file, method)
        segments = split_by_joystick(actions, joysticks, rdot, phidot, obstacles)
        print(f"Found {len(segments)} segments in {file}")

        timestamp = os.path.splitext(file)[0]
        video_path = os.path.join(output_folder, f"{folder}_{method}_{timestamp}.mp4")
        create_experiment_video(segments, video_path, method=method)
        print(f"Saved video: {video_path}")

print("All experiment videos created!")
