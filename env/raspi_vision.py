import depthai as dai
import gin
from gymnasium import Wrapper
import torch
from models.autoencoder import AutoEncoder

gin.parse_config_file("config/settings.gin")
from config.config import Config

config = Config()

from controller.camera_thread import CameraThread


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
    manip.initialConfig.setResize(config.image_size, config.image_size)
    manip.setKeepAspectRatio(True)

    # Link camera preview output to ImageManip input
    cam_rgb.preview.link(manip.inputImage)

    # Optionally, create an XLinkOut to stream the grayscale image to the host
    xout = pipeline.createXLinkOut()
    xout.setStreamName("rgb")
    manip.out.link(xout.input)
    return dai.Device(pipeline)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RaspiImageWrapper(Wrapper):
    def __init__(self, env, image_every=1):
        super().__init__(env=env)
        self.device = create_pipeline()
        self.autoencoder = AutoEncoder(
            (3, config.image_size, config.image_size), config.n_features
        ).to(device)
        self.autoencoder.load_state_dict(
            torch.load("autoencoder.pth", map_location=device)
        )
        self.camera_thread = CameraThread(
            camera=self.device, encoder=self.autoencoder, fps=config.fps
        )
        self.camera_thread.start()

    def step(self, action):
        s, r, d, t, i = self.env.step(action)
        self.features = self.camera_thread.get_latest()
        i["features"] = self.features
        return s, r, d, t, i

    def reset(self, **kwargs):
        s, i = self.env.reset(**kwargs)
        self.features = self.camera_thread.get_latest()
        i["features"] = self.features
        return s, i
