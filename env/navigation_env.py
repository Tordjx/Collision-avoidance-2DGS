import importlib
import os
import sys
from collections import deque

import gymnasium as gym
import numpy as np
import pinocchio as pin
import torch
from scipy.spatial.transform import Rotation

from autoencoder import AutoEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def import_all_modules_from_package(package_name):
    splatting_dir = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../third_party/2d-gaussian-splatting",
        )
    )
    if splatting_dir not in sys.path:
        sys.path.append(splatting_dir)
    package_dir = os.path.join(splatting_dir, package_name)
    for module_name in os.listdir(package_dir):
        if module_name.endswith(".py") and module_name != "__init__.py":
            module_name = module_name[:-3]  # Remove the .py extension
            importlib.import_module(f"{package_name}.{module_name}")


import_all_modules_from_package("gaussian_renderer")
import_all_modules_from_package("scene")
import_all_modules_from_package("utils")
import cv2
import torchvision.transforms as transforms
from gaussian_renderer import GaussianModel, render
from scene.cameras import Camera
from torchvision.transforms import functional as F
from upkie_description import load_in_pinocchio


def euler_to_quaternion(euler_angles):
    r = Rotation.from_euler("xyz", euler_angles)
    quaternion = r.as_quat()
    return quaternion


class NavigationEnv(gym.Env):
    def __init__(
        self, max_duration=10, dt=1 / 10, window=True, eval=False, no_encoder=False
    ):
        super(NavigationEnv, self).__init__()
        self.eval = eval
        self.action_space = gym.spaces.Box(low=-2, high=2, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-10, high=10, shape=(2 + 2 + 32,), dtype=np.float32
        )
        self.position = np.zeros(3)  # SE(2): (x, y, theta)
        self.velocity = np.zeros(2)  # (linear velocity, yaw velocity)
        self.dt = dt
        self.max_duration = int(max_duration / self.dt)
        self.timestep = 0
        self.joystick = np.zeros(2)
        self.robot_height = 0.5
        # Data augmentation
        self.saturation = 0.5
        self.brightness = 0.3
        self.contrast = 0.5
        self.hue = 0.4
        self.noise_std = 0.01
        self.data_augmentation = transforms.Compose(
            [
                transforms.Lambda(
                    lambda img: F.adjust_brightness(
                        img,
                        torch.empty(1)
                        .uniform_(1 - self.brightness, 1 + self.brightness)
                        .item(),
                    )
                ),
                transforms.Lambda(
                    lambda img: F.adjust_contrast(
                        img,
                        torch.empty(1)
                        .uniform_(1 - self.contrast, 1 + self.contrast)
                        .item(),
                    )
                ),
                transforms.Lambda(
                    lambda img: F.adjust_hue(
                        img, torch.empty(1).uniform_(-self.hue, self.hue).item()
                    )
                ),
                transforms.Lambda(
                    lambda img: F.adjust_saturation(
                        img,
                        torch.empty(1)
                        .uniform_(1 - self.saturation, 1.5 + self.saturation)
                        .item(),
                    )
                ),
                transforms.Lambda(
                    lambda img: torch.clamp(
                        img + torch.randn_like(img) * self.noise_std, 0, 1
                    )
                ),
            ]
        )

        # Collision detection
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
        self.tilt = 0
        # Gaussian splatting
        self.gaussians = GaussianModel(3)
        fovx = 69 * np.pi / 180
        fovy = 54 * np.pi / 180
        model_path = "data"
        self.gaussians.load_ply(os.path.join(model_path, "point_cloud.ply"))
        self.fovx, self.fovy = fovx, fovy
        self.znear, self.zfar = 0.02, 100
        self.width = 128
        self.height = 128
        self.window = window
        if window:
            self.window_name = str(np.random.random())
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # Autoencoder
        self.encoder = AutoEncoder(
            input_shape=(3, self.height, self.width), z_size=32
        ).to(device)
        if not no_encoder:
            self.encoder.load_state_dict(
                torch.load("autoencoder.pth", map_location=device)
            )
        self.encoder.eval()
        self.features_memory = deque(maxlen=1)
        # Position history
        self.total_timesteps = 0
        self.position_history = []

    def step(self, action):
        x, y, theta = self.position
        self.position_history.append([x, y, theta, self.total_timesteps])
        self.total_timesteps += 1
        if self.total_timesteps % 10000 == 0 and not self.eval:
            np.save("position_history.npy", np.array(self.position_history))
        j_y, j_x = -self.joystick
        self.tilt += 0.02 * np.random.randn()
        self.tilt = np.clip(self.tilt, -np.pi / 4, np.pi / 4)
        # Note: velocity is in SE(2), position is (x, y, theta)-
        action[1] *= -1
        corrected = np.clip(np.array([j_x, j_y]) + action, -1, 1)
        self.velocity = (
            np.clip(
                corrected, self.velocity - self.dt * 1.2, self.velocity + self.dt * 1.2
            )
            + 0.05 * np.random.randn()
        )
        self.velocity[0] = np.clip(self.velocity[0], -1.5, 1.5)
        self.velocity[1] = np.clip(self.velocity[1], -1, 1)  # angular velocity !
        velocity_cartesian = self.velocity[0] * np.array(
            [np.cos(self.position[2]), np.sin(self.position[2])]
        )
        self.position[0] += velocity_cartesian[0] * self.dt
        self.position[1] += velocity_cartesian[1] * self.dt
        self.position[2] += self.velocity[1] * self.dt  # Yaw update
        image = self.query_image()
        with torch.no_grad():
            image = (
                self.data_augmentation(torch.from_numpy(image))
                .to(device)
                .to(torch.float32)
            )
            self.features_memory.append(self.encoder.encode(image).cpu().numpy())
        self.timestep += 1
        trunc = self.timestep >= self.max_duration
        done = self.compute_done()
        if done:
            reward = -100
        else:
            reward = 1 - np.sum(abs(action))  # self.reward()

        if np.random.binomial(1, 1 / (10 / self.dt)):
            self.joystick = self.sample_joystick()
        observation = self.get_obs()
        info = {}
        return observation, float(reward), bool(done), bool(trunc), info

    def get_obs(self):
        features = np.concatenate(np.array(self.features_memory), -1).flatten()

        observation = np.concatenate([self.velocity, self.joystick, features])
        return observation.astype(np.float32)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.robot_height = np.random.uniform(0.4, 0.6)
        self.tilt = 0
        self.position = self.sample_position()  # Reset (x, y, theta)
        while self.compute_distance() < 0.5:
            self.position = self.sample_position()
        self.velocity = np.zeros(2)  # Reset (linear velocity, yaw velocity)
        self.timestep = 0
        self.joystick = self.sample_joystick()
        image = self.query_image()
        for _ in range(1):
            with torch.no_grad():
                image = (
                    self.data_augmentation(torch.from_numpy(image))
                    .to(device)
                    .to(torch.float32)
                )
                self.features_memory.append(self.encoder.encode(image).cpu().numpy())
        observation = self.get_obs()
        return observation, {}

    def sample_position(self):
        if np.random.choice([0, 1]):
            x, y = np.random.uniform(
                low=np.array([-1.7, -3.8]),
                high=np.array([2.8, 6.4]),
                size=2,
            )
        else:
            x, y = np.random.uniform(
                low=np.array([2.8, -3.8]),
                high=np.array([12.1, 1.3]),
                size=2,
            )
        return np.array([x, y, np.random.uniform(-np.pi, np.pi)])

    def sample_joystick(self):
        if self.eval:
            return np.array([0, -1])
        else:
            if np.random.choice([0, 1]):
                return np.random.uniform(-1, 1, size=2)
            else:
                return np.random.choice([-1, 0, 1], size=2)

    def query_image(self):
        posx, posy, theta = self.position
        x, y, z, w = euler_to_quaternion([0, self.tilt, theta])
        C = np.array([posx, posy, self.robot_height])
        R = Rotation.from_quat([x, y, z, w]).as_matrix()
        R = (
            R
            @ Rotation.from_quat(
                euler_to_quaternion([np.pi / 2, np.pi, np.pi / 2])
            ).as_matrix()
        )
        T = -R.T @ C
        view = Camera(
            colmap_id=0,
            R=R,
            T=T,
            FoVx=self.fovx,
            FoVy=self.fovy,
            image=torch.zeros((3, self.width, self.height)),
            gt_alpha_mask=None,
            image_name="",
            uid=0,
        )

        class PipelineParams:
            def __init__(self):
                self.convert_SHs_python = False
                self.compute_cov3D_python = False
                self.debug = False
                self.depth_ratio = 0.0

        pipeline = PipelineParams()
        bg_color = [1, 1, 1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            rendering = render(view, self.gaussians, pipeline, background)
            image = rendering["render"].permute(1, 2, 0).cpu().numpy()
            depth = rendering["surf_depth"].cpu().numpy()
            self.image = (
                self.data_augmentation(torch.from_numpy(image).moveaxis(-1, 0))
                .to(torch.float32)
                .numpy()
            )
            self.depth = depth
        if self.window:
            window_image = (255 * image).astype(np.uint8)
            upscaled_image = cv2.resize(
                window_image, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_LINEAR
            )
            upscaled_image = cv2.cvtColor(upscaled_image, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.window_name, upscaled_image)
            cv2.waitKey(1)
        return np.moveaxis(image, -1, 0)

    def compute_distance(self):
        joints = [
            "left_hip",
            "left_knee",
            "left_wheel",
            "right_hip",
            "right_knee",
            "right_wheel",
        ]
        posx, posy, theta = self.position
        orientation = euler_to_quaternion([0, 0, theta])
        x, y, z, w = orientation
        q = np.array(
            [
                posx,
                posy,
                self.robot_height,
                x,
                y,
                z,
                w,
            ]
            + [0 for joint in joints]
        )
        data = self.model.createData()
        geom_data = pin.GeometryData(self.collision_model)
        pin.computeDistances(self.model, data, self.collision_model, geom_data, q)
        distance = np.min([x.min_distance for x in geom_data.distanceResults])
        return distance

    def compute_done(self):
        distance = self.compute_distance()
        return distance < 0.1
