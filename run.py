import gymnasium as gym
import numpy as np
import torch
import upkie.envs

upkie.envs.register()
import gin

gin.parse_config_file(f"config/settings.gin")
from config.settings import EnvSettings

# from robot_state_randomization import RobotStateRandomization
env_settings = EnvSettings()
from env.envs import make_vision_pink_env

gym.envs.registration.register(
    id="UpkieServos-v5", entry_point="env.upkie_servos:UpkieServos"
)


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        agent_frequency = env_settings.agent_frequency
        max_episode_duration = 25000

        velocity_env = gym.make(
            env_settings.env_id,
            max_episode_steps=int(max_episode_duration * agent_frequency),
            frequency=agent_frequency,
            regulate_frequency=False,
            shm_name="upkie",
            # max_ground_velocity=env_settings.max_ground_velocity,
            spine_config=env_settings.spine_config,
            fall_pitch=np.pi / 2,
            # no_imu = env_settings.no_imu
        )
        env = make_vision_pink_env(
            velocity_env,
            env_settings,
            eval_mode=False,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


device = torch.device("cuda")

# env setup


envs = make_env(env_settings.env_id, 0, 0, 0, "")()
from sb3_contrib import CrossQ
from tqdm import tqdm

model = CrossQ.load("CrossQ_navigation", env=envs)
obs, infos = envs.reset()
smooth_action = 0
for i in tqdm(range(200000)):

    action, _ = model.predict(obs, deterministic=True)
    smooth_action = action
    obs, r, d, t, infos = envs.step(smooth_action)
    if infos["spine_observation"]["joystick"]["triangle_button"]:
        obs, infos = envs.reset()
    if d:
        smooth_action = 0
        obs, infos = envs.reset()
