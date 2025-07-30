import depthai as dai
import cv2
import numpy as np

class RaysRaspi:
    def __init__(self): 
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
        self.N = 20
        self.radius = 5

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
