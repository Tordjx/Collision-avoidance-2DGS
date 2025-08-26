import numpy as np
from statsmodels.stats.proportion import proportions_ztest

# Success rates (%) from the table
foa_success = np.array([100, 100, 30, 100, 80, 0, 0, 90])
rl_success  = np.array([100, 70, 100, 100, 70, 20, 20, 70])

# Number of trials per experiment
n_trials = 10

# Convert percentages to counts of successes
foa_counts = (foa_success / 100 * n_trials).astype(int)
rl_counts  = (rl_success  / 100 * n_trials).astype(int)

envs = ["Cart 1", "Cart 2", "Cart 3", "Cart 4", 
        "Corridor (straight)", "Corridor (sideways)", 
        "120° L", "Server"]

print("=== Two-proportion z-tests (H0: FOA > RL) ===")
for env, s0, s1 in zip(envs, foa_counts, rl_counts):
    stat, pval = proportions_ztest([s0, s1], [n_trials, n_trials], alternative="smaller")
    print(f"{env:20s}: FOA={s0}/{n_trials}, RL={s1}/{n_trials}, z={stat:.2f}, p={pval:.4f}")

print("=== Two-proportion z-tests (H0: FOA < RL) ===")
for env, s0, s1 in zip(envs, foa_counts, rl_counts):
    stat, pval = proportions_ztest([s0, s1], [n_trials, n_trials], alternative="larger")
    print(f"{env:20s}: FOA={s0}/{n_trials}, RL={s1}/{n_trials}, z={stat:.2f}, p={pval:.4f}")
# Aggregate across all experiments
total_foa_success = foa_counts.sum()
total_rl_success  = rl_counts.sum()
total_trials = len(foa_counts) * n_trials  # total per method

print("Aggregated across all experiments:")
print(f"FOA successes: {total_foa_success}/{total_trials}")
print(f"RL successes : {total_rl_success}/{total_trials}")

# Two-proportion z-test, one-sided FOA > RL
stat, pval = proportions_ztest([total_foa_success, total_rl_success],
                               [total_trials, total_trials],
                               alternative="smaller")
print(f"\nTwo-proportion z-test (H1: FOA > RL): z={stat:.2f}, p={pval:.4f}")

# Two-proportion z-test, one-sided FOA < RL
stat, pval = proportions_ztest([total_foa_success, total_rl_success],
                               [total_trials, total_trials],
                               alternative="larger")
print(f"Two-proportion z-test (H1: FOA < RL): z={stat:.2f}, p={pval:.4f}")
