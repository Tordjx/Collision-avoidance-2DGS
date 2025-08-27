import os
import time
import logging
import gymnasium as gym
import numpy as np
import torch
import upkie.envs
import gin
import csv
from upkie.utils.robot_state import RobotState
from config.settings import EnvSettings
from env.envs import make_vision_pink_env
from env.robot_state_randomization import RobotStateRandomization
from sb3_contrib import CrossQ
from tqdm import tqdm
import pinocchio as pin
from upkie_description import load_in_pinocchio
from scipy.spatial.transform import Rotation

# --- Register env ---
upkie.envs.register()
gym.envs.registration.register(
    id="UpkieServos-v5", entry_point="env.upkie_servos:UpkieServos"
)

# --- Load settings ---
gin.parse_config_file("config/settings.gin")
env_settings = EnvSettings()

def euler_to_quaternion(euler_angles):
    r = Rotation.from_euler("xyz", euler_angles)
    quaternion = r.as_quat()
    return quaternion

class CollisionWrapper(gym.Wrapper):
    def __init__(self, env):
        super(CollisionWrapper, self).__init__(env)
        self.robot = load_in_pinocchio(
            root_joint=pin.JointModelFreeFlyer(), variant="camera"
        )
        env_model, env_collision_model, env_visual_model = pin.buildModelsFromUrdf(
            "data/manual_postprocess.urdf", "data"
        )
        self.model, collision_model = pin.appendModel(
            env_model,
            self.robot.model,
            env_collision_model,
            self.robot.collision_model,
            pin.WORLD,
            pin.SE3.Identity(),
        )
        env_id = collision_model.getGeometryId("baseLink_0")
        for i in range(collision_model.ngeoms):
            if not i == env_id:
                collision_pair = pin.CollisionPair(i, env_id)
                collision_model.addCollisionPair(collision_pair)
        self.collision_model = collision_model

    def step(self, action):
        s, r, d, t, i = self.env.step(action)
        d = d or self.compute_distance(i) < 0.02
        return s, r, d, t, i

    def reset(self, **kwargs):
        s, i = self.env.reset(**kwargs)
        while self.compute_distance(i) < 0.5:
            s, i = self.env.reset(**kwargs)
        return s, i

    def compute_distance(self, info):
        joints = [
            "left_hip",
            "left_knee",
            "left_wheel",
            "right_hip",
            "right_knee",
            "right_wheel",
        ]
        posx, posy, posz = info['spine_observation']['sim']['base']['position']
        w, x, y, z = info['spine_observation']['sim']['base']['orientation']
        q = np.array(
            [
                posx,
                posy,
                posz,
                x,
                y,
                z,
                w,
            ] + [0 for joint in joints]
        )
        data = self.model.createData()
        geom_data = pin.GeometryData(self.collision_model)
        pin.computeDistances(self.model, data, self.collision_model, geom_data, q)
        distance = np.min([x.min_distance for x in geom_data.distanceResults])
        return distance

# --- Logger setup ---
def setup_logger(name):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    filename = os.path.join("logs", f"{name}_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(filename, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    return logger

logger = setup_logger("eval")

spine_config = {
    "bullet": {
        "torque_control": {
            "kd": 0.3,
            "kp": 400.0
        }
    }
}

# --- Env factory ---
def make_env(seed=0):
    agent_frequency = env_settings.agent_frequency
    max_episode_duration = 50

    base_env = gym.make(
        env_settings.env_id,
        max_episode_steps=int(max_episode_duration * agent_frequency),
        frequency=100,
        regulate_frequency=False,
        shm_name="upkie",
        spine_config=spine_config,
        fall_pitch=np.pi / 3,
        init_state=RobotState(randomization=RobotStateRandomization()),
    )
    env = make_vision_pink_env(base_env, env_settings, eval_mode=True)
    env = CollisionWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    return env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CSV setup ---
os.makedirs("results", exist_ok=True)
csv_file = os.path.join("results", "evaluation_results_no_policy.csv")
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["agent_name", "episode_idx", "episode_length"])

    # --- Evaluation loop ---

    env = make_env(seed=0)

    for ep in tqdm(range(100)):
        obs, info = env.reset()
        done, truncated = False, False
        steps = 0

        while not (done or truncated):
            action, _ = np.zeros(2)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1

        # log to CSV
        writer.writerow(["", ep + 1, steps])
        logger.info(f"{""} | Episode {ep+1} length: {steps}")

    env.close()

logger.info(f"Results saved to {csv_file}")
