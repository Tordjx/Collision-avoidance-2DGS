#%%
import matplotlib 
matplotlib.use("Agg")     # if you just save plots to file, no GUI
import trimesh
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from env.navigation_env import NavigationEnv
from sb3_contrib import CrossQ
from tqdm import tqdm

### BOUNDARIES
# Load mesh
mesh = trimesh.load("/home/vtordjma/GoPro/manual_postprocess.obj")

# Filter faces where all vertices are in z ∈ [zmin, zmax] and x < 13
zmin, zmax = 0.0, 10
face_vertices_z = mesh.vertices[mesh.faces][:, :, 2]
face_vertices_x = mesh.vertices[mesh.faces][:, :, 0]
faces_mask = (face_vertices_z >= zmin) & (face_vertices_z <= zmax) & (face_vertices_x < 13)
faces_mask = faces_mask.all(axis=1)

# Keep only faces in slice
faces = mesh.faces[faces_mask]
vertices = mesh.vertices.copy()
vertices_2d = vertices[:, :2]  # flatten z
triangles = [vertices_2d[f] for f in faces]

# Create PolyCollection for boundaries
collection = PolyCollection(triangles, facecolor='k', edgecolor='k', linewidths=0.5)

### RUN ENV + MODEL
env = NavigationEnv(eval=True, window = False)
import gymnasium as gym
env = gym.wrappers.TimeLimit(env, max_episode_steps=15)
model = CrossQ.load("CrossQ_navigation", env=env)
s, i = env.reset()
positions, actions = [], []

for _ in tqdm(range(100000)):
    action, _ = model.predict(s, deterministic=True)
    action_cart = np.array([action[0], action[1]*0.3]) #VARIGNON
    actions.append(action_cart)
    positions.append(i['position'])
    s, r, d, t, i = env.step(action)
    if d:
        s, i = env.reset()

positions = np.array(positions)
actions = np.array(actions)
np.save("positions.npy", positions)
np.save("actions.npy", actions)
### PROCESS ACTIONS
xs = positions[:, 0]
ys = positions[:, 1]
phis = positions[:, 2]

v_forward = actions[:, 0]   # forward correction in robot frame
omega     = actions[:, 1]   # angular correction

# Rotate forward correction into world frame
u = v_forward * np.cos(phis)
v = v_forward * np.sin(phis)

# Correction norm (combine forward + angular)
corr_norm = np.sqrt(v_forward**2 + omega**2)

# Normalize to [0,1] for colormap and alpha
normed = (corr_norm - corr_norm.min()) / (corr_norm.max() - corr_norm.min() + 1e-8)

# Use colormap
cmap = plt.cm.coolwarm
colors = cmap(normed)
colors[:, -1] = normed  # set alpha = strength

### FINAL PLOT
fig, ax = plt.subplots(figsize=(10, 10))

# Boundaries
ax.add_collection(collection)

# Quiver plot of actions
ax.quiver(xs, ys, u, v, color=colors, angles="xy", scale_units="xy", scale=1)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")
ax.autoscale()  # fit to mesh + actions
ax.set_title("Action corrections in world frame\n(color & opacity = correction norm)")

plt.savefig('boundaries_arrows.pdf')

# %%
