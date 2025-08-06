from upkie.utils.raspi import configure_agent_process, on_raspi
if on_raspi() : 
    configure_agent_process()
from tqdm import tqdm
from foa.foa import ReactiveAvoidance
import numpy as np
from foa.envs import make_rays_pink_env

import gymnasium as gym
import numpy as np
import torch
import upkie.envs

upkie.envs.register()
import gin

gin.parse_config_file(f"config/settings.gin")
from config.settings import EnvSettings

# from robot_state_randomization import RobotStateRandomization
from upkie.utils.robot_state_randomization import RobotStateRandomization
from upkie.utils.robot_state import RobotState
env_settings = EnvSettings()
from env.envs import make_vision_pink_env

gym.envs.registration.register(
    id="UpkieServos-v5", entry_point="env.upkie_servos:UpkieServos"
)


reactive_avoidance = ReactiveAvoidance(control_radius=0.15)
def modulate_velocity(reactive_avoidance, i ):
    target_forward = -i["spine_observation"]["joystick"]["left_axis"][1]
    target_yaw = -i["spine_observation"]["joystick"]["left_axis"][0]
    reference_velocity = np.array([target_forward,target_yaw])
    obstacle_points = i["obstacle_points"]
    modulated_velocity = reactive_avoidance.compute(reference_velocity, obstacle_points)
    modulated_velocity[1] *=-1
    correction = modulated_velocity - reference_velocity
    return correction

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
    # no_imu = env_settings.no_imu,
    init_state = RobotState(position_base_in_world = np.array([2,2,0.58]))
)
env = make_rays_pink_env(
    velocity_env,
    env_settings,
    eval_mode=False,
)
s,i = env.reset()
for _ in tqdm(range(200000)):
    action = modulate_velocity(reactive_avoidance, i)
    s,r,d,t,i = env.step(action)
    if i["spine_observation"]["joystick"]["triangle_button"]:
        obs, i = env.reset()
    if d:
        obs, i = env.reset()
