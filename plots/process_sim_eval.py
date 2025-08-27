import pandas as pd 
import numpy as np

df_seeds = pd.read_csv("/home/vtordjma/Collision-avoidance-2DGS/results/evaluation_results.csv")
df_no_policy = pd.read_csv("/home/vtordjma/Collision-avoidance-2DGS/results/evaluation_results_no_policy.csv")

df_seeds = df_seeds[df_seeds['episode_length']>10]
df_no_policy = df_no_policy[df_no_policy['episode_length']>10]

lengths_policies = df_seeds['episode_length'].values/10
lengths_no_policy = df_no_policy['episode_length'].values/10

print("With policy: mean length = {:.2f}, std = {:.2f}".format(np.mean(lengths_policies), np.std(lengths_policies)))
print("Without policy: mean length = {:.2f}, std = {:.2f}".format(np.mean(lengths_no_policy), np.std(lengths_no_policy)))