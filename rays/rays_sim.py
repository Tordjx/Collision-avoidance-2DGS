import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from gaussian_renderer import GaussianModel, render
from scene.cameras import Camera


class MinimalDepthEnv:
    def __init__(self):
        # Load Gaussian splatting model and point cloud
        self.gaussians = GaussianModel(3)
        model_path = "data"
        self.gaussians.load_ply(os.path.join(model_path, "point_cloud.ply"))

        # Camera parameters
        self.fovx = 69 * (np.pi / 180)
        self.fovy = 54 * (np.pi / 180)
        self.znear = 0.02
        self.zfar = 100
        self.width = 128
        self.height = 128

        # Ray shooting params
        self.N = 20  # Number of rays horizontally
        self.radius = 2  # Patch half-size for median filtering

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def render_depth(self, position, quaternion):
        """
        Render depth image from given position and quaternion.

        position: array-like, shape (3,) - x, y, z (height)
        quaternion: array-like, shape (4,) - [x, y, z, w] quaternion (xyzw convention)
        """
        posx, posy, posz = position
        x, y, z, w = quaternion

        C = [posx, posy, posz]
        R = Rotation.from_quat([x, y, z, w]).as_matrix()

        # Adjust camera orientation to match rendering conventions
        adjust_rot = Rotation.from_euler("xyz", [np.pi / 2, np.pi, np.pi / 2]).as_matrix()
        R = R @ adjust_rot

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
        background = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            rendering = render(view, self.gaussians, pipeline, background)
            depth = rendering["surf_depth"].cpu().numpy()

        return depth

    def get_rays(self, position, quaternion):
        """
        Given a 3D position and orientation quaternion, render depth and
        return median depth values on a horizontal slice (row) of the depth image.
        """
        depth = self.render_depth(position, quaternion)

        frame_height, frame_width = depth.shape

        # Focal length in pixels (approximate)
        f_y = frame_height / (2 * np.tan(self.fovy / 2))
        c_y = frame_height / 2

        # Compute pixel row corresponding to camera optical axis elevation:
        # We find vertical angle of the camera direction vector to image plane center
        # The elevation angle = arcsin of the z-axis of rotation matrix
        R = Rotation.from_quat(quaternion).as_matrix()
        camera_z_axis = R[:, 2]  # camera's forward vector
        elevation = np.arcsin(camera_z_axis[2])  # vertical angle of optical axis

        pixel_row = int(np.clip(f_y * np.tan(-elevation) + c_y, 0, frame_height - 1))

        # Precompute circular mask for median filtering
        Y, X = np.ogrid[-self.radius : self.radius + 1, -self.radius : self.radius + 1]
        circular_mask = X**2 + Y**2 <= self.radius**2

        cols = np.linspace(0, frame_width - 1, self.N, dtype=int)

        median_depths = []
        for col in cols:
            # Define patch boundaries
            row_start = max(pixel_row - self.radius, 0)
            row_end = min(pixel_row + self.radius + 1, frame_height)
            col_start = max(col - self.radius, 0)
            col_end = min(col + self.radius + 1, frame_width)

            patch = depth[row_start:row_end, col_start:col_end]

            # Adjust mask if patch is smaller at edges
            mask_r_start = self.radius - (pixel_row - row_start)
            mask_r_end = mask_r_start + patch.shape[0]
            mask_c_start = self.radius - (col - col_start)
            mask_c_end = mask_c_start + patch.shape[1]

            patch_mask = circular_mask[mask_r_start:mask_r_end, mask_c_start:mask_c_end]

            # Apply mask and compute median of valid pixels inside circle
            median_val = np.median(patch[patch_mask])
            median_depths.append(float(median_val))

        return median_depths, pixel_row, cols
    def get_ray_points_local_frame(self, position, quaternion):
        """
        Return 3D points (in meters) in the robot's local frame (x forward, y left, z up)
        corresponding to rays across the rendered depth image.
        """
        depth = self.render_depth(position, quaternion)
        median_depths, pixel_row, cols = self.get_rays(position, quaternion)

        # Intrinsics
        fx = self.width / (2 * np.tan(self.fovx / 2))
        fy = self.height / (2 * np.tan(self.fovy / 2))
        cx = self.width / 2
        cy = self.height / 2

        points = []
        for u, depth_val in zip(cols, median_depths):
            v = pixel_row

            # From image coordinates (u, v) + depth to camera coordinates (OpenCV: z forward)
            z_cam = depth_val
            x_cam = (u - cx) * z_cam / fx
            y_cam = (v - cy) * z_cam / fy

            # Convert to robot frame: x forward, y left, z up
            x_robot = z_cam
            y_robot = -x_cam
            z_robot = -y_cam

            points.append(np.array([x_robot, y_robot, z_robot]))

        return np.array(points)
