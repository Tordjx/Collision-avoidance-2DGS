import numpy as np
import depthai as dai
import open3d as o3d
import threading
from loop_rate_limiters import RateLimiter

class ObstaclePointCloud:
    def __init__(self, fps=10, max_points=30, threshold=0.2, voxel_size=100.0):
        self.fps = fps
        self.dt = 1.0 / fps
        self.max_points = max_points
        self.threshold = threshold
        self.voxel_size = voxel_size
        self.lock = threading.Lock()
        self.latest_points = None
        self.running = False
        self.pitch = 0.0

        self.pipeline = self._create_pipeline()
        self.device = dai.Device(self.pipeline)
        self.queue = self.device.getOutputQueue(name="out", maxSize=4, blocking=False)

        self.thread = threading.Thread(target=self.run)
        self.rate_limiter = RateLimiter(frequency=fps)

    def _create_pipeline(self):
        pipeline = dai.Pipeline()
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        depth = pipeline.create(dai.node.StereoDepth)
        pointcloud = pipeline.create(dai.node.PointCloud)
        sync = pipeline.create(dai.node.Sync)
        xOut = pipeline.create(dai.node.XLinkOut)

        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoLeft.setCamera("left")
        monoLeft.setFps(30)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoRight.setCamera("right")
        monoRight.setFps(30)

        depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        depth.setLeftRightCheck(True)
        depth.setExtendedDisparity(False)
        depth.setSubpixel(True)

        config = depth.initialConfig.get()
        config.postProcessing.speckleFilter.enable = True
        config.postProcessing.speckleFilter.speckleRange = 28
        config.postProcessing.temporalFilter.enable = False
        config.postProcessing.spatialFilter.enable = False
        config.postProcessing.thresholdFilter.minRange = 150
        config.postProcessing.thresholdFilter.maxRange = 5000
        config.postProcessing.decimationFilter.decimationFactor = 1
        depth.initialConfig.set(config)

        monoLeft.out.link(depth.left)
        monoRight.out.link(depth.right)
        depth.depth.link(pointcloud.inputDepth)

        pointcloud.outputPointCloud.link(sync.inputs["pcl"])
        xOut.setStreamName("out")
        sync.out.link(xOut.input)
        xOut.input.setBlocking(False)

        return pipeline

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def run(self):
        while self.running:
            inMessage = self.queue.tryGet()
            if inMessage is not None and "pcl" in inMessage:
                pclData = inMessage["pcl"]
                points = pclData.getPoints().astype(np.float64)

                if points is not None and len(points) > 0:
                    transformed = self._transform_to_robot_frame(points, self.pitch)
                    filtered = transformed[transformed[:, 2] > self.threshold]
                    downsampled = self._downsample_points(filtered)

                    with self.lock:
                        self.latest_points = downsampled

            self.rate_limiter.sleep()

    def set_pitch(self, pitch_rad):
        self.pitch = pitch_rad

    def get_latest(self):
        with self.lock:
            return self.latest_points

    def _transform_to_robot_frame(self, points_pc: np.ndarray, pitch_rad: float) -> np.ndarray:
        if points_pc is None or len(points_pc) == 0:
            return np.empty((0, 3), dtype=np.float64)

        points_swapped = np.empty_like(points_pc)
        points_swapped[:, 0] = points_pc[:, 2]
        points_swapped[:, 1] = -points_pc[:, 1]
        points_swapped[:, 2] = points_pc[:, 0]

        c = np.cos(pitch_rad)
        s = np.sin(pitch_rad)
        R_y = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])

        return points_swapped @ R_y.T

    def _downsample_points(self, points: np.ndarray) -> np.ndarray:
        if points is None or len(points) == 0:
            return np.empty((0, 3))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        downsampled = np.asarray(pcd_down.points)
        if len(downsampled) > self.max_points:
            idx = np.random.choice(len(downsampled), self.max_points, replace=False)
            downsampled = downsampled[idx]
        return downsampled
