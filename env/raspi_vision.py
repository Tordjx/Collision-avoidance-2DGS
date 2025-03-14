import depthai as dai
import gin
import gymnasium as gym
import numpy as np
from gymnasium import Wrapper

from Public_repo.config.settings import EnvSettings, TrainingSettings

gin.parse_config_file("settings.gin")
train_settings = TrainingSettings()
env_settings = EnvSettings()


def create_pipeline():
    # Create the pipeline
    pipeline = dai.Pipeline()

    # Define a color camera
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(640, 480)  # Original resolution
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # Define Image Manipulation for cropping and grayscale conversion
    manip = pipeline.createImageManip()

    # Configure resizing settings
    manip.initialConfig.setResize(env_settings.height, env_settings.width)
    manip.setKeepAspectRatio(False)

    # Link camera preview output to ImageManip input
    cam_rgb.preview.link(manip.inputImage)

    # Optionally, create an XLinkOut to stream the grayscale image to the host
    xout = pipeline.createXLinkOut()
    xout.setStreamName("rgb")
    manip.out.link(xout.input)
    return dai.Device(pipeline)


def get_image(device):
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    in_rgb = None
    while in_rgb is None:
        # Get RGB frames
        in_rgb = q_rgb.tryGet()
        # If we have a new RGB frame, process it
        if in_rgb is not None:
            frame = in_rgb.getCvFrame() / 255
    return frame


class RaspiImageWrapper(Wrapper):
    def __init__(self, env, image_every=1):
        super().__init__(env=env)
        self.device = create_pipeline()
        self.image_every = image_every
        self.image_count = 0
        low = self.env.observation_space.low
        self.observation_space = gym.spaces.Box(
            low=low,
            high=np.concatenate([env.observation_space.high]),
            shape=low.shape,
            dtype=env.observation_space.dtype,
        )

    def step(self, action):

        s, r, d, t, i = self.env.step(action)
        if self.image_count % self.image_every == 0:
            image = get_image(self.device)
            image = image[..., ::-1]
            self.image = image
        else:
            image = self.image
        i["image"] = np.moveaxis(image, -1, 0)
        self.image_count += 1
        return s, r, d, t, i

    def reset(self, **kwargs):
        s, i = self.env.reset(**kwargs)
        image = get_image(self.device)
        self.image = image
        i["image"] = np.moveaxis(image, -1, 0)
        self.image_count = 0
        return s, i
