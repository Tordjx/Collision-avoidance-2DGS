import argparse

parser = argparse.ArgumentParser(description="Process dataset parameters.")
parser.add_argument(
    "--max_episode_duration", type=int, help="Episode duration in seconds", default=10
)
parser.add_argument(
    "--number_episodes", type=int, help="Number of episodes", default=10
)

args = parser.parse_args()
from gymnasium.vector import SyncVectorEnv
from sb3_contrib import CrossQ

from env.navigation_env import NavigationEnv

envs = SyncVectorEnv(
    [NavigationEnv(window=True, eval=True, max_duration=args.max_episode_duration)]
)
model = CrossQ.load("CrossQ_navigation", env=envs)

for i in range(args.number_episodes):
    d = False
    t = False
    obs, infos = envs.reset()
    while not d or t:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, d, t, infos = envs.step(action)
