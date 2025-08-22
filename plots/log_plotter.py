import matplotlib.pyplot as plt
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
folders = os.listdir(log_folder)
folders = [x for x in folders if "CAM" not in x]
folders = [x for x in folders if "output" not in x]

def parse_experiment(filepath):
    """Parse joystick, action, rdot, phidot from a log file."""
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

                rdot_val = float(match_rdot.group(1))
                phidot_val = float(match_phidot.group(1))

                actions.append(action)
                joystick.append(joy)
                rdot.append(rdot_val)
                phidot.append(phidot_val)

    return np.array(actions), np.array(joystick), np.array(rdot), np.array(phidot)


# --- choose one experiment file to plot ---
folder = folders[0]  # pick first folder (adjust as needed)
files = os.listdir(os.path.join(log_folder, folder))
experiment_file = os.path.join(log_folder, folder, files[0])  # pick first file

actions, joysticks, rdot, phidot = parse_experiment(experiment_file)

# --- plotting raw signals ---
fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

axs[0].plot(actions[:, 0], label="Action[0]")
axs[0].plot(actions[:, 1], label="Action[1]")
axs[0].set_ylabel("Action")
axs[0].legend()

axs[1].plot(joysticks[:, 0], label="Joystick[0]")
axs[1].plot(joysticks[:, 1], label="Joystick[1]")
axs[1].set_ylabel("Joystick")
axs[1].legend()

axs[2].plot(rdot, label="rdot")
axs[2].set_ylabel("rdot")
axs[2].legend()

axs[3].plot(phidot, label="phidot")
axs[3].set_ylabel("phidot")
axs[3].set_xlabel("Timestep")
axs[3].legend()

plt.tight_layout()
plt.savefig("raw_log_plot.png")


def split_by_joystick(actions, joysticks, rdot, phidot, target=[0.001, -1.0], tol=1e-6):
    segments = []
    start_idx = None

    def matches(j):
        return np.allclose(j, target, atol=tol)

    for i, joy in enumerate(joysticks):
        if matches(joy):
            if start_idx is None:
                start_idx = i
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

    segments = [x for x in segments if len(x['actions']) > 20]  # filter too short
    return segments


def plot_segment_traj(segment):
    x, y, phi = 0, 0, 0
    dt = 0.1
    X, Y, phis = [], [], []
    commanded_velocities = []
    modulated_velocities = []
    actions = segment['actions']
    joysticks = segment['joysticks']
    rdots = segment['rdot']
    phidots = segment['phidot']
    for i in range(len(rdots)):
        X.append(x)
        Y.append(y)
        phis.append(phi)
        rdot = rdots[i]
        phidot = phidots[i]

        commanded_vel = np.array([
            -joysticks[i][1]*np.cos(phi + dt*-joysticks[i][0]),
            -joysticks[i][1]*np.sin(phi + dt*-joysticks[i][0])
        ])
        mod_joystick = actions[i] + np.array([-joysticks[i][0], -joysticks[i][1]])
        modulated_vel = np.array([
            mod_joystick[1]*np.cos(phi + dt*mod_joystick[0]),
            mod_joystick[1]*np.sin(phi + dt*mod_joystick[0])
        ])
        commanded_velocities.append(commanded_vel)
        modulated_velocities.append(modulated_vel)

        x,y, phi = integrate(x,y,phi, rdot, phidot)

    return X, Y, phis, commanded_velocities, modulated_velocities


def plot_segments_trajectories(segments, folder, experiment_file, arrow_stride=10, show=True):
    plt.figure(figsize=(8, 8))

    for idx, segment in enumerate(segments):
        X, Y, phis, commanded_vels, modulated_vels = plot_segment_traj(segment)
        X = np.array(X)
        Y = np.array(Y)
        commanded_vels = np.array(commanded_vels)
        modulated_vels = np.array(modulated_vels)

        plt.plot(X, Y, label=f"Trajectory {idx+1}", alpha=0.7)

        # clean arrows using quiver
        plt.quiver(
            X[::arrow_stride], Y[::arrow_stride],
            commanded_vels[::arrow_stride, 0], commanded_vels[::arrow_stride, 1],
            color='blue', alpha=0.5, scale=20, width=0.003,
            label="Commanded vel" if idx == 0 else ""
        )
        plt.quiver(
            X[::arrow_stride], Y[::arrow_stride],
            modulated_vels[::arrow_stride, 0], modulated_vels[::arrow_stride, 1],
            color='red', alpha=0.5, scale=20, width=0.003,
            label="Modulated vel" if idx == 0 else ""
        )

    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.axis("equal")

    method = "foa" if "foa" in experiment_file else "rl"
    timestamp = os.path.basename(experiment_file).split(method)[-1][1:]
    plt.title(f"Experiment: {folder} | Method: {method.upper()} | Timestamp: {timestamp}")

    plt.legend()
    plt.grid(True)

    if show:
        plt.show()
    else : 
        plt.savefig(f"/home/vtordjma/Documents/log_expe_icra/output/trajectory_plot_{folder}_{method}_{timestamp}.png")


# --- use functions ---
for folder in folders :
    files = os.listdir(os.path.join(log_folder, folder))
    for file in files : 
        experiment_file = os.path.join(log_folder, folder, file)  # pick first file

        actions, joysticks, rdot, phidot = parse_experiment(experiment_file)
        segments = split_by_joystick(actions, joysticks, rdot, phidot)

        print(f"Found {len(segments)} repetitions")
        for i, seg in enumerate(segments):
            print(f"Segment {i}: length={len(seg['actions'])}")

        plot_segments_trajectories(segments, folder, experiment_file, arrow_stride=5, show = False)
