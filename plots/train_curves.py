import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

log_root = "../rl_train"  # <-- path to the folder containing your seeds
tag = "rollout/ep_len_mean"  # <-- change this to the scalar you want to plot
"rollout/ep_rew_mean"
def find_event_file(root):
    """Find first TensorBoard event file under a root directory."""
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.startswith("events.out.tfevents"):
                return os.path.join(dirpath, f)
    return None

def load_tensorboard_scalars(path, tag):
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values

# Collect all seeds
all_steps, all_values = [], []
for seed_dir in os.listdir(log_root):
    seed_path = os.path.join(log_root, seed_dir)
    if not os.path.isdir(seed_path):
        continue
    event_file = find_event_file(seed_path)
    if event_file:
        steps, values = load_tensorboard_scalars(event_file, tag)
        all_steps.append(steps)
        all_values.append(values)
        print(f"Loaded {event_file} with {len(values)} points")

# Align runs
min_len = min(len(v) for v in all_values)
common_steps = all_steps[0][:min_len]
aligned = np.stack([v[:min_len] for v in all_values], axis=0)

# Compute mean & std
mean = aligned.mean(axis=0)
std = aligned.std(axis=0)

# Plot
plt.figure(figsize=(8,5))
plt.plot(common_steps, mean)#, label="My RL method", color="C0")
plt.fill_between(common_steps, mean-std, mean+std, alpha=0.2, color="C0")

plt.xlabel("Environment steps")
plt.ylabel("Episode length")
plt.title(f"Evaluation across {len(all_values)} seeds")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('train_curve_len.pdf', dpi=150)
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

log_root = "../rl_train"  # <-- path to the folder containing your seeds
tag = "rollout/ep_rew_mean"  # <-- change this to the scalar you want to plot

def find_event_file(root):
    """Find first TensorBoard event file under a root directory."""
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.startswith("events.out.tfevents"):
                return os.path.join(dirpath, f)
    return None

def load_tensorboard_scalars(path, tag):
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values

# Collect all seeds
all_steps, all_values = [], []
for seed_dir in os.listdir(log_root):
    seed_path = os.path.join(log_root, seed_dir)
    if not os.path.isdir(seed_path):
        continue
    event_file = find_event_file(seed_path)
    if event_file:
        steps, values = load_tensorboard_scalars(event_file, tag)
        all_steps.append(steps)
        all_values.append(values)
        print(f"Loaded {event_file} with {len(values)} points")

# Align runs
min_len = min(len(v) for v in all_values)
common_steps = all_steps[0][:min_len]
aligned = np.stack([v[:min_len] for v in all_values], axis=0)

# Compute mean & std
mean = aligned.mean(axis=0)
std = aligned.std(axis=0)

# Plot
plt.figure(figsize=(8,5))
plt.plot(common_steps, mean)#, label="My RL method", color="C0")
plt.fill_between(common_steps, mean-std, mean+std, alpha=0.2, color="C0")

plt.xlabel("Environment steps")
plt.ylabel("Episode return")
plt.title(f"Evaluation across {len(all_values)} seeds")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('train_curve_rew.pdf', dpi=150)
