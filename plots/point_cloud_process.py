import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
import time

def generate_sphere(center, radius, num_points):
    """Generate points on the surface of a sphere visible from the camera."""
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    theta = np.random.uniform(0, np.pi / 2, num_points)  # Limit to front hemisphere
    x = center[0] + radius * np.sin(theta) * np.cos(phi)
    y = center[1] + radius * np.sin(theta) * np.sin(phi)
    z = center[2] + radius * np.cos(theta)
    return np.vstack((x, y, z)).T

def generate_cube(center, size, num_points):
    """Generate points on the surface of a cube visible from the camera."""
    x = np.random.uniform(center[0] - size / 2, center[0] + size / 2, num_points)
    y = np.random.uniform(center[1] - size / 2, center[1] + size / 2, num_points)
    z = np.random.uniform(center[2], center[2] + size / 2, num_points)  # Only front-facing
    return np.vstack((x, y, z)).T

def add_noise_and_outliers(points, noise_level, outlier_ratio):
    """Add Gaussian noise and random outliers to the point cloud."""
    noisy_points = points + np.random.normal(0, noise_level, points.shape)
    num_outliers = int(outlier_ratio * len(points))
    outliers = np.random.uniform(-10, 10, (num_outliers, 3))  # Random outliers
    outliers = outliers[outliers[:, 2] > 0]  # Ensure outliers are in front of the camera
    return np.vstack((noisy_points, outliers))

def visualize_point_cloud_open3d(points):
    """Visualize the point cloud using Open3D."""
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([point_cloud])

objects = []
for i in range(10):  # Add 10 spheres
    sphere = generate_sphere(center=np.random.uniform(-10, 10, 3), radius=np.random.uniform(1, 3), num_points=1000)
    objects.append(sphere)

for i in range(10):  # Add 10 cubes
    cube = generate_cube(center=np.random.uniform(-10, 10, 3), size=np.random.uniform(1, 3), num_points=1000)
    objects.append(cube)

# Combine all objects into a single point cloud
point_cloud = np.vstack(objects)
point_cloud_noisy = add_noise_and_outliers(point_cloud, noise_level=0.1, outlier_ratio=0.1)
# Filter points to ensure they are visible from the camera
point_cloud_noisy = point_cloud_noisy[point_cloud_noisy[:, 2] > 0]
def remove_outliers_and_downsample(points, nb_neighbors=20, std_ratio=0.1):
    """Remove outliers using statistical outlier removal."""
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    clean_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    uni_down_pcd = clean_cloud.uniform_down_sample(every_k_points=points.shape[0]//20)
    return np.asarray(uni_down_pcd.points)
def remove_outliers(points, nb_neighbors=20, std_ratio=0.1):
    """Remove outliers using statistical outlier removal."""
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    clean_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return np.asarray(clean_cloud.points)
# Remove outliers from the noisy point cloud
filtered_point_cloud = remove_outliers(point_cloud_noisy)
def visualize_point_clouds_matplotlib(noisy_points, filtered_points, downsampled_points):
    """Visualize the noisy, filtered, and downsampled point clouds in separate subplots using Matplotlib."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 10))

    # Determine the limits for the axes
    all_points = np.vstack((noisy_points, filtered_points, downsampled_points))
    x_limits = (np.min(all_points[:, 0]), np.max(all_points[:, 0]))
    y_limits = (np.min(all_points[:, 1]), np.max(all_points[:, 1]))
    z_limits = (np.min(all_points[:, 2]), np.max(all_points[:, 2]))

    # Noisy points subplot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(noisy_points[:, 0], noisy_points[:, 1], noisy_points[:, 2], 
                c='red', alpha=0.25, label='Noisy Points')
    ax1.set_title("Noisy Points")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim(x_limits)
    ax1.set_ylim(y_limits)
    ax1.set_zlim(z_limits)
    ax1.legend()

    # Filtered points subplot
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2], 
                c='green', alpha=0.5, label='Filtered Points')
    ax2.set_title("Filtered Points")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim(x_limits)
    ax2.set_ylim(y_limits)
    ax2.set_zlim(z_limits)
    ax2.legend()

    # Downsampled points subplot
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(downsampled_points[:, 0], downsampled_points[:, 1], downsampled_points[:, 2], 
                c='blue', alpha=1, label='Downsampled Points')
    ax3.set_title("Downsampled Points")
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_xlim(x_limits)
    ax3.set_ylim(y_limits)
    ax3.set_zlim(z_limits)
    ax3.legend()

    plt.suptitle("Point Clouds Visualization")
    plt.tight_layout()
    plt.show()

# Downsample the filtered point cloud
filtered_point_cloud = remove_outliers(point_cloud_noisy)
downsampled_point_cloud = remove_outliers_and_downsample(point_cloud_noisy)

# Plot the point clouds
visualize_point_clouds_matplotlib(point_cloud_noisy, filtered_point_cloud, downsampled_point_cloud)
import matplotlib.pyplot as plt

# Plot noisy and filtered point clouds with larger points and good alpha
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot noisy points
ax.scatter(point_cloud_noisy[:, 0], point_cloud_noisy[:, 1], point_cloud_noisy[:, 2], 
           c='red', alpha=0.2, s=1, label='Noisy Points')

# Plot filtered points with larger size and higher alpha
ax.scatter(downsampled_point_cloud[:, 0], downsampled_point_cloud[:, 1], downsampled_point_cloud[:, 2], 
           c='green', alpha=0.8, s=200, label='Filtered  and downsampled Points')

ax.set_title("Noisy vs Filtered Point Cloud")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()
def benchmark_filter_and_downsample(point_cloud, trials=100):
    """Benchmark the filtering and downsampling process over multiple trials."""
    filter_times = []
    downsample_times = []

    for _ in range(trials):
        # Measure filtering time
        start_time = time.time()
        filtered_cloud = remove_outliers(point_cloud)
        filter_times.append(time.time() - start_time)

        # Measure downsampling time
        start_time = time.time()
        downsampled_cloud = remove_outliers_and_downsample(point_cloud)
        downsample_times.append(time.time() - start_time)

    avg_filter_time = np.mean(filter_times)
    avg_downsample_time = np.mean(downsample_times)

    print(f"Average filtering time over {trials} trials: {avg_filter_time:.6f} seconds")
    print(f"Average downsampling time over {trials} trials: {avg_downsample_time:.6f} seconds")

# Run the benchmark
benchmark_filter_and_downsample(point_cloud_noisy)