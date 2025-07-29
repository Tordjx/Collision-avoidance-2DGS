import depthai as dai
import cv2
import numpy as np
# Create pipeline
pipeline = dai.Pipeline()

# Mono cameras
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Stereo depth node
stereo = pipeline.createStereoDepth()
#stereo.setConfidenceThreshold(245)  # High confidence
stereo.setRectifyEdgeFillColor(10000)   # Black, for consistency
stereo.setLeftRightCheck(True)      # Better handling of occlusions
stereo.setExtendedDisparity(True)   # Subpixel matching with wider range
stereo.setSubpixel(True)            # Better depth precision
#stereo.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)  # Reduce noise

# Link mono to stereo
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# Output depth
xoutDepth = pipeline.createXLinkOut()
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)
# Parameters for ray shooting
N = 20  # number of rays horizontally
window_size = 5  # median filter window half-size (window height = 2*window_size + 1)
theta_deg = -15  # camera pitch angle downward in degrees (adjust this!)
theta = np.deg2rad(theta_deg)

# Approximate camera intrinsics (you should replace with your real calibration!)
frame_height = 480  # mono camera resolution height
frame_width = 640  # width is usually 640 for OAK-D Lite mono cams
f_y = 365  # focal length in pixels (approximate)
c_y = frame_height / 2  # principal point y (approximate)

def get_pixel_row_for_theta(theta, f_y, c_y):
    # pixel row corresponding to vertical angle alpha = -theta
    # alpha = arctan((y - c_y)/f_y)
    # => y = f_y * tan(alpha) + c_y
    y = f_y * np.tan(-theta) + c_y
    return int(np.clip(y, 0, frame_height - 1))
import numpy as np

# radius of circular window (in pixels)
radius = 5

# Precompute circular mask
Y, X = np.ogrid[-radius:radius+1, -radius:radius+1]
circular_mask = X**2 + Y**2 <= radius**2  # boolean mask

with dai.Device(pipeline) as device:
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        inDepth = depthQueue.get()
        depthFrame = inDepth.getFrame().astype(np.float32)
        depthFrame = np.where(depthFrame== 0, 1e4, depthFrame)  # replace 0 with a large value (1e4)
        depthFrame = np.clip(depthFrame, 0, 1e4)

        pixel_row = get_pixel_row_for_theta(theta, f_y, c_y)

        cols = np.linspace(0, frame_width - 1, N, dtype=int)

        median_depths = []
        for col in cols:
            # define patch boundaries
            row_start = max(pixel_row - radius, 0)
            row_end = min(pixel_row + radius + 1, frame_height)
            col_start = max(col - radius, 0)
            col_end = min(col + radius + 1, frame_width)

            patch = depthFrame[row_start:row_end, col_start:col_end]

            # adjust mask if patch is smaller at edges
            mask_r_start = radius - (pixel_row - row_start)
            mask_r_end = mask_r_start + patch.shape[0]
            mask_c_start = radius - (col - col_start)
            mask_c_end = mask_c_start + patch.shape[1]

            patch_mask = circular_mask[mask_r_start:mask_r_end, mask_c_start:mask_c_end]

            # apply mask and compute median of valid pixels inside circle
            median_val = np.median(patch[patch_mask])
            median_depths.append(float(median_val)/1e3)

        print(f"Median depths at row {pixel_row} with radius {radius}: {median_depths}")

        # Visualization (existing code)
        depthFrameVis = np.log(depthFrame + 10)
        depthFrameVis = 255 * (depthFrameVis - np.log(10)) / (np.log(1e4 + 10) - np.log(10))
        depthFrameVis = cv2.convertScaleAbs(depthFrameVis)
        depthColor = cv2.applyColorMap(depthFrameVis, cv2.COLORMAP_JET)

        cv2.line(depthColor, (0, pixel_row), (frame_width - 1, pixel_row), (0, 255, 255), 2)
        for col in cols:
            cv2.circle(depthColor, (col, pixel_row), 5, (255, 255, 255), 1)

        cv2.imshow("Depth - Jet Colormap with Rays", depthColor)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
