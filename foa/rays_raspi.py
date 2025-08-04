import depthai as dai
import cv2
import numpy as np
import gymnasium as gym
class RaysRaspiWrapper(gym.Wrapper) : 
    def __init__(self, env,image_every = 10):
        super(RaysRaspiWrapper, self).__init__(env)
        self.rays_raspi = RaysRaspi()
        self.image_every = image_every
        self.obstacle_points = None
        self.image_count = 0
    def reset(self, **kwargs):
        s,i = self.env.reset(**kwargs)
        self.image_count = 0
        i['obstacle_points'] = self.rays_raspi.get_ray_points_local_frame(abs(i['spine_observation']['base_orientation']['pitch']))
        return s,i
    def step(self, action):
        s,r,d,t,i = self.env.step(action)
        if self.image_count % self.image_every == 0:
            i['obstacle_points']= self.rays_raspi.get_ray_points_local_frame(abs(i['spine_observation']['base_orientation']['pitch']))
            self.obstacle_points = i['obstacle_points']
        else:
            i['obstacle_points'] = self.obstacle_points
        self.image_count += 1
        return s,r,d,t,i
class RaysRaspi:
    def __init__(self, N = 100): 
        # Frame size
        self.frame_height = 480
        self.frame_width = 640
        
        # Create pipeline
        self.pipeline = dai.Pipeline()

        # Mono cameras
        monoLeft = self.pipeline.createMonoCamera()
        monoRight = self.pipeline.createMonoCamera()
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # Stereo depth node
        stereo = self.pipeline.createStereoDepth()
        stereo.setRectifyEdgeFillColor(10000)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(True)
        stereo.setSubpixel(True)

        # Link mono to stereo
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        # Output depth
        xoutDepth = self.pipeline.createXLinkOut()
        xoutDepth.setStreamName("depth")
        stereo.depth.link(xoutDepth.input)

        # Parameters for ray shooting
        self.N = N
        self.radius = 5 #pixels

    def get_rays(self, theta, depthFrame):
        fovx = 69 * np.pi / 180
        fovy = 54 * np.pi / 180
        f_y = self.frame_height / (2 * np.tan(fovy / 2))
        c_y = self.frame_height / 2

        def get_pixel_row_for_theta(theta, f_y, c_y):
            y = f_y * np.tan(-theta) + c_y
            return int(np.clip(y, 0, self.frame_height - 1))

        pixel_row = get_pixel_row_for_theta(theta, f_y, c_y)
        cols = np.linspace(0, self.frame_width - 1, self.N, dtype=int)

        # Precompute circular mask
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

        return median_depths, pixel_row, cols

    def vizualize(self, median_depths, depthFrame, pixel_row, cols):
        # Prepare visualization image
        depthFrameVis = np.log(depthFrame + 10)
        depthFrameVis = 255 * (depthFrameVis - np.log(10)) / (np.log(1e4 + 10) - np.log(10))
        depthFrameVis = cv2.convertScaleAbs(depthFrameVis)
        depthColor = cv2.applyColorMap(depthFrameVis, cv2.COLORMAP_JET)

        # Draw ray line
        cv2.line(depthColor, (0, pixel_row), (self.frame_width - 1, pixel_row), (0, 255, 255), 2)

        # Draw each ray sample point and its depth value
        for col, depth in zip(cols, median_depths):
            cv2.circle(depthColor, (col, pixel_row), 5, (255, 255, 255), 1)
            label = f"{depth:.2f}m"
            cv2.putText(depthColor, label, (col + 5, pixel_row - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Show the image
        cv2.imshow("Depth - Jet Colormap with Rays", depthColor)

    def visualize_camera(self, theta=0.0):
        with dai.Device(self.pipeline) as device:
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

            while True:
                inDepth = depthQueue.get()
                depthFrame = inDepth.getFrame().astype(np.float32)
                depthFrame = np.where(depthFrame == 0, 1e4, depthFrame)
                depthFrame = np.clip(depthFrame, 0, 1e4)

                median_depths, pixel_row, cols = self.get_rays(theta, depthFrame)
                self.vizualize(median_depths, depthFrame, pixel_row, cols)

                if cv2.waitKey(1) == ord('q'):
                    break

        cv2.destroyAllWindows()
    def get_ray_points_local_frame(self, theta):
        """
        Return 3D points in the robot's local frame.
        Robot frame: x forward, y left, z up/down (camera frame assumed z forward, x right, y down)
        """
        with dai.Device(self.pipeline) as device:
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

            while True:
                inDepth = depthQueue.get()
                depthFrame = inDepth.getFrame().astype(np.float32)
                depthFrame = np.where(depthFrame == 0, 1e4, depthFrame)
                depthFrame = np.clip(depthFrame, 0, 1e4)
        median_depths, pixel_row, cols = self.get_rays(theta, depthFrame)

        # Intrinsics
        fovx = 69 * np.pi / 180
        fovy = 54 * np.pi / 180
        fx = self.frame_width / (2 * np.tan(fovx / 2))
        fy = self.frame_height / (2 * np.tan(fovy / 2))
        cx = self.frame_width / 2
        cy = self.frame_height / 2

        points = []
        for u, depth in zip(cols, median_depths):
            v = pixel_row

            # Convert (u,v,depth) to camera coordinates (OpenCV: z forward, x right, y down)
            z_cam = depth
            x_cam = (u - cx) * z_cam / fx
            y_cam = (v - cy) * z_cam / fy

            # Convert camera to robot frame (robot x forward, y left, z up/down)
            x_robot = z_cam
            y_robot = -x_cam
            z_robot = -y_cam

            points.append(np.array([x_robot, y_robot]))

        return np.array(points).T
