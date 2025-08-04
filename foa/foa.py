import numpy as np
from vartools.states import ObjectPose
from fast_obstacle_avoidance.obstacle_avoider import SampledClusterAvoider
import numpy as np
from vartools.states import ObjectPose
import gymnasium as gym

class MinimalRobot2D:
    """Simple 2D robot with fixed LIDAR at origin and circular control radius."""

    def __init__(self, control_radius=0.15, control_point=np.array([0.0, 0.0])):
        self.pose = ObjectPose(position=np.zeros(2), orientation=0.0)

        self.control_radius = control_radius
        self.control_point = control_point  # Relative to robot base

    @property
    def rotation_matrix(self):
        theta = self.pose.orientation
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
        ])

    def transform_to_world(self, points):
        """Transform points from robot to world frame."""
        return self.rotation_matrix @ points.T + self.pose.position[:, None]

    def transform_to_robot(self, points):
        """Transform points from world to robot frame."""
        return self.rotation_matrix.T @ (points.T - self.pose.position[:, None])

class ReactiveAvoidance:
    def __init__(self, control_radius=0.4):
        # Setup robot model
        self.robot = MinimalRobot2D()
        self.robot.control_radius = control_radius
        self.robot.control_point = [0, 0]

        # Avoider instance
        self.avoider = SampledClusterAvoider(control_radius=self.robot.control_radius)

    def compute(self, reference_velocity, obstacle_points):
        """Returns modulated velocity given obstacle points and reference velocity.
        
        Parameters:
        - reference_velocity: np.array of shape (2,) – desired velocity before obstacle avoidance
        - obstacle_points: np.array of shape (N, 2) – sampled points (e.g., raycast or lidar)

        Returns:
        - modulated_velocity: np.array of shape (2,)
        """

        self.robot.pose.orientation = 0.0  # Optional: update if orientation is relevant

        # Update avoider with obstacle points
        self.avoider.update_laserscan(obstacle_points, in_robot_frame=False)
        print(reference_velocity)
        # Modulate the velocity
        modulated_velocity = self.avoider.avoid(
            reference_velocity,
            self.robot.pose.position
        )

        return modulated_velocity
