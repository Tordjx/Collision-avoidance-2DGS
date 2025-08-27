from upkie.utils.raspi import configure_agent_process, on_raspi
if on_raspi() : 
    configure_agent_process()
    reg_freq = False
else :
    reg_freq = False
import gymnasium as gym
import numpy as np
import torch
import upkie.envs
from loop_rate_limiters import RateLimiter
upkie.envs.register()
import gin
from env.robot_state_randomization import RobotStateRandomization
gin.parse_config_file(f"config/settings.gin")
from config.settings import EnvSettings

# from robot_state_randomization import RobotStateRandomization
from upkie.utils.robot_state import RobotState
env_settings = EnvSettings()
from env.envs import make_vision_pink_env

gym.envs.registration.register(
    id="UpkieServos-v5", entry_point="env.upkie_servos:UpkieServos"
)

import logging
import time
import os
def setup_logger(method_name):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join("logs", f"{method_name}_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any old handlers
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(filename, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    return filename

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        agent_frequency = env_settings.agent_frequency
        max_episode_duration = 25000

        velocity_env = gym.make(
            env_settings.env_id,
            max_episode_steps=int(max_episode_duration * agent_frequency),
            frequency=100,
            regulate_frequency=reg_freq,
            shm_name="upkie",
            # max_ground_velocity=env_settings.max_ground_velocity,
            spine_config=env_settings.spine_config,
            fall_pitch=np.pi / 2,
            init_state = RobotState(randomization = RobotStateRandomization()),
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
rate_limiter = RateLimiter(frequency=10)

envs = make_env(env_settings.env_id, 0, 0, 0, "")()
from sb3_contrib import CrossQ
from tqdm import tqdm
log_file = setup_logger("rl")
logging.info(f"Starting run, logging to {log_file}")
model = CrossQ.load("CrossQ_navigation", env=envs)
obs, infos = envs.reset()
smooth_action = 0
alpha= 0.5
for i in tqdm(range(200000)):
    rate_limiter.sleep()
    if infos['spine_observation']['joystick']['left_axis'][1] >= 0:
        action = np.zeros(2)
    else: 
        action, _ = model.predict(obs, deterministic=True)
    smooth_action = alpha * smooth_action + (1 - alpha) * action
    obs, r, d, t, infos = envs.step(smooth_action)
    if infos["spine_observation"]["joystick"]["triangle_button"]:
        obs, infos = envs.reset()
    if d:
        smooth_action = 0
        obs, infos = envs.reset()
    joystick_input = infos["spine_observation"]["joystick"]['left_axis']

    forward_velocity = infos["spine_observation"]["wheel_odometry"]["velocity"]
    yaw_velocity = infos["spine_observation"]["base_orientation"][
        "angular_velocity"
    ][2]
    logging.info(
        f"Action={action}, Joystick={joystick_input}, rdot={forward_velocity}, phidot={yaw_velocity} "
    )