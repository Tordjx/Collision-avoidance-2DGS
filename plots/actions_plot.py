import os
import re
import numpy as np
import csv
import matplotlib.pyplot as plt

log_folder = "/home/vtordjma/Documents/log_expe_icra"
folders = [f for f in os.listdir(log_folder) if "CAM" not in f and "output" not in f and "resultats" not in f]

output_folder = os.path.join(log_folder, "output_actions")
os.makedirs(output_folder, exist_ok=True)


def parse_experiment(filepath):
    actions, joysticks, rdot, phidot = [], [], [], []
    with open(filepath, "r") as f:
        for line in f:
            ma = re.search(r"Action=\[(.*?)\], Joystick", line)
            mj = re.search(r"Joystick=\[(.*?)\], rdot", line)
            mr = re.search(r"rdot=([-\d.eE]+)", line)
            mp = re.search(r"phidot=([-\d.eE]+)", line)
            if ma and mj and mr and mp:
                actions.append([float(v) for v in ma.group(1).split()])
                joysticks.append([float(v.replace(",", "")) for v in mj.group(1).split()])
                rdot.append(float(mr.group(1)))
                phidot.append(float(mp.group(1)))
    return np.array(actions), np.array(joysticks), np.array(rdot), np.array(phidot)


def split_by_joystick(actions, joysticks, rdot, phidot, target=[0.001, -1.0], tol=1e-6, min_len=0):
    """Return list of (start_idx, end_idx) segments instead of just lengths."""
    segments, start_idx = [], None
    def matches(j): return np.allclose(j, target, atol=tol)

    for i, joy in enumerate(joysticks):
        if matches(joy):
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                if i - start_idx > min_len:
                    segments.append((start_idx, i))
                start_idx = None

    if start_idx is not None and len(actions) - start_idx > min_len:
        segments.append((start_idx, len(actions)))

    return segments


# --- collect data ---
results = []

for folder in folders:
    files = os.listdir(os.path.join(log_folder, folder))
    for file in files:
        experiment_file = os.path.join(log_folder, folder, file)
        actions, joysticks, rdot, phidot = parse_experiment(experiment_file)
        segments = split_by_joystick(actions, joysticks, rdot, phidot)
        if not segments:
            continue

        # Get experiment info
        method = "foa" if "foa" in file.lower() else "rl"
        timestamp = re.findall(r"\d+", file)[-1] if re.findall(r"\d+", file) else "NA"

        # Save plots
        # Save plots
        for seg_id, (start, end) in enumerate(segments):
            seg_actions = actions[start:end]

            mean_dim1 = np.mean(seg_actions[:, 1])

            plt.figure()
            plt.plot(seg_actions[:, 0], label="Action dim 0")
            plt.plot(seg_actions[:, 1], label="Action dim 1")
            plt.xlabel("Timestep")
            plt.ylabel("Action value")
            plt.title(f"{method.upper()} | {folder} | {file} | Segment {seg_id}\nMean(dim1)={mean_dim1:.3f}")
            plt.legend()

            seg_output_folder = os.path.join(output_folder, method, folder)
            os.makedirs(seg_output_folder, exist_ok=True)
            out_path = os.path.join(seg_output_folder, f"{os.path.splitext(file)[0]}_seg{seg_id}.png")
            plt.savefig(out_path, dpi=150)
            plt.close()


        mean_length_sec = np.mean([end - start for (start, end) in segments]) * 0.1  # timestep=0.1s
        mean_len = len(segments)
        results.append([folder, file, method, timestamp, mean_length_sec, mean_len])

# --- save CSV ---
csv_file = os.path.join(log_folder, "segments_mean_length.csv")
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Experiment", "File", "Method", "Timestamp", "MeanSegmentLength_s", "len"])
    writer.writerows(results)

print(f"CSV saved to {csv_file}")
print(f"Plots saved in {output_folder}")
