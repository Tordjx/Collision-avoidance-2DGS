import os
import re
import numpy as np
import csv

log_folder = "/home/vtordjma/Documents/log_expe_icra"
folders = [f for f in os.listdir(log_folder) if "CAM" not in f and "output" not in f and "resultats" not in f]

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

def split_by_joystick(actions, joysticks, rdot, phidot, target=[0.001, -1.0], tol=1e-6, min_len=20):
    segments, start_idx = [], None
    def matches(j): return np.allclose(j, target, atol=tol)
    for i, joy in enumerate(joysticks):
        if matches(joy):
            if start_idx is None: start_idx = i
        else:
            if start_idx is not None:
                if i - start_idx > min_len:
                    segments.append(i - start_idx)
                start_idx = None
    if start_idx is not None and len(actions) - start_idx > min_len:
        segments.append(len(actions) - start_idx)
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
        mean_length_sec = np.mean(segments) * 0.1  # timestep = 0.1s
        mean_len = len(segments)
        method = "foa" if "foa" in file else "rl"
        timestamp = re.findall(r"\d+", file)[-1]  # last number in filename
        results.append([folder, file, method, timestamp, mean_length_sec, mean_len])

# --- save CSV ---
csv_file = os.path.join(log_folder, "segments_mean_length.csv")
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Experiment", "File", "Method", "Timestamp", "MeanSegmentLength_s", "len"])
    writer.writerows(results)

print(f"CSV saved to {csv_file}")
