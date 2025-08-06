import threading
import gin
import numpy as np
import torch
from loop_rate_limiters import RateLimiter

gin.parse_config_file("config/settings.gin")
from config.settings import EnvSettings

config = EnvSettings()


def get_image(device):
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    in_rgb = None
    while in_rgb is None:
        # Get RGB frames
        in_rgb = q_rgb.tryGet()
        # If we have a new RGB frame, process it
        if in_rgb is not None:
            frame = in_rgb.getCvFrame() / 255
    return torch.from_numpy(frame.astype(np.float32)).moveaxis(-1, 0)


class CameraThread:
    def __init__(self, camera, encoder, fps=10):
        """
        camera: your camera interface object, e.g. with .get_image()
        encoder: your encoder object or function
        fps: desired frame rate
        """
        self.camera = camera
        self.encoder = encoder
        self.fps = fps
        self.dt = 1.0 / fps
        self.rate_limiter = RateLimiter(frequency = fps)
        self.latest_encoded = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = threading.Thread(target=self.run)

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def run(self):
        while self.running:
            # Get image from camera
            img = get_image(self.camera)
            
            
            # Encode image
            with torch.no_grad():
                encoded = self.encoder.encode(img).squeeze(0).numpy()

            # Store safely
            with self.lock:
                self.latest_encoded = encoded

            self.rate_limiter.sleep()

    def get_latest(self):
        with self.lock:
            return self.latest_encoded
