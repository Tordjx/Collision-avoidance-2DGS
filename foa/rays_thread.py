import threading
import numpy as np
import torch
from loop_rate_limiters import RateLimiter
import depthai as dai


class RaysThread:
    def __init__(self, N=20, fps=10):
        self.N = N
        self.fps = fps
        self.dt = 1.0 / fps
        self.frame_height = 480
        self.frame_width = 640
        self.radius = 5
        self.lock = threading.Lock()
        self.latest_points = None
        self.running = False

        # DepthAI pipeline
        self.pipeline = dai.Pipeline()

        monoLeft = self.pipeline.createMonoCamera()
        monoRight = self.pipeline.createMonoCamera()
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        stereo = self.pipeline.createStereoDepth()
        stereo.setRectifyEdgeFillColor(10000)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(True)
        stereo.setSubpixel(True)

        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        xoutDepth = self.pipeline.createXLinkOut()
        xoutDepth.setStreamName("depth")
        stereo.depth.link(xoutDepth.input)

        self.device = dai.Device(self.pipeline)
        self.depthQueue = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        self.thread = threading.Thread(target=self.run)
        self.rate_limiter = RateLimiter(frequency=fps)

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def run(self):
        while self.running:
            inDepth = self.depthQueue.tryGet()
            if inDepth is not None:
                frame = inDepth.getFrame().astype(np.float32)
                frame = np.where(frame == 0, 1e4, frame)
                frame = np.clip(frame, 0, 1e4)
                pitch = 0.0  # default fallback, can be externally set
                points = self._get_ray_points_local_frame(pitch, frame)

                with self.lock:
                    self.latest_points = points

            self.rate_limiter.sleep()

    def set_pitch(self, pitch):
        self.pitch = pitch

    def get_latest(self):
        with self.lock:
            return self.latest_points

    def _get_ray_points_local_frame(self, theta, depthFrame):
        fovx = 69 * np.pi / 180
        fovy = 54 * np.pi / 180
        fx = self.frame_width / (2 * np.tan(fovx / 2))
        fy = self.frame_height / (2 * np.tan(fovy / 2))
        cx = self.frame_width / 2
        cy = self.frame_height / 2

        f_y = self.frame_height / (2 * np.tan(fovy / 2))
        c_y = self.frame_height / 2
        pixel_row = int(np.clip(f_y * np.tan(-theta) + c_y, 0, self.frame_height - 1))
        cols = np.linspace(0, self.frame_width - 1, self.N, dtype=int)

        Y, X = np.ogrid[-self.radius:self.radius+1, -self.radius:self.radius+1]
        circular_mask = X**2 + Y**2 <= self.radius**2

        median_depths = []
        for col in cols:
            row_start = max(pixel_row - self.radius, 0)
            row_end = min(pixel_row + self.radius + 1, self.frame_height)
            col_start = max(col - self.radius, 0)
            col_end = min(col + self.radius + 1, self.frame_width)

            patch = depthFrame[row_start:row_end, col_start:col_end]

            mask_r_start = self.radius - (pixel_row - row_start)
            mask_r_end = mask_r_start + patch.shape[0]
            mask_c_start = self.radius - (col - col_start)
            mask_c_end = mask_c_start + patch.shape[1]

            patch_mask = circular_mask[mask_r_start:mask_r_end, mask_c_start:mask_c_end]
            median_val = np.median(patch[patch_mask])
            median_depths.append(float(median_val) / 1e3)

        points = []
        for u, depth in zip(cols, median_depths):
            v = pixel_row
            z_cam = depth
            x_cam = (u - cx) * z_cam / fx
            y_cam = (v - cy) * z_cam / fy
            x_robot = z_cam
            y_robot = -x_cam
            points.append(np.array([x_robot, y_robot]))

        return np.array(points).T  # shape (2, N)
