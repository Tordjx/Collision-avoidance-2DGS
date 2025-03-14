import argparse

parser = argparse.ArgumentParser(description="Process dataset parameters.")
parser.add_argument(
    "--len_dataset", type=int, help="Number of samples in the dataset", default=60000
)

args = parser.parse_args()
import torch
from tqdm import tqdm

from env.navigation_env import NavigationEnv

env = NavigationEnv(window=False, eval=True, max_duration=5, no_encoder=True)
images = []
depths = []
s, i = env.reset()
for i in tqdm(range(args.len_dataset)):
    images.append(torch.from_numpy(env.image))
    depths.append(torch.from_numpy(env.depth))
    s, r, d, t, i = env.step(env.action_space.sample())
    if d or t:
        s, i = env.reset()

images_tensor = torch.stack(images)
depths_tensor = torch.stack(depths)
torch.save(images_tensor, "images.pt")
torch.save(depths_tensor, "depths.pt")
