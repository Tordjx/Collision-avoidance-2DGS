
# Collision avoidance from monocular vision trained with novel view synthesis
This repository contains the code for the working paper Collision avoidance from monocular vision trained with novel view synthesis, available on HAL https://hal.science/hal-05005146.
## Installation

### Clone the Repository

First, clone the repository recursively to ensure all submodules are included:

```bash
git clone https://github.com/Tordjx/Collision-avoidance-2DGS.git --recursive
```

### Conda Environment

Add the following line to `third_party/2d-gaussian-splatting/submodules/simple-knn/simple_knn.cu` (source: [PyTorch Audio PR #3811](https://github.com/pytorch/audio/pull/3811/files)):

```cpp
#include <float.h>
```

Apply a similar patch to `third_party/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h:

```cpp
#include <cstdint>
```

Now, create the `vision_agent` conda environment:

```bash
conda env create -f environment.yaml
conda activate vision_agent
```

## Creating a GS Dataset
### Camera Settings

To ensure high-quality image captures for accurate 3D reconstruction, follow these camera settings:

1. **Shutter Speed**: Set the shutter speed to at least 1/125s to avoid motion blur. This is crucial for maintaining clear images, particularly when capturing in dynamic environments.
2. **ISO & Aperture**: Adjust the ISO and aperture to avoid underexposure, especially in indoor settings. A larger aperture (lower f-number) allows more light, while a higher ISO setting compensates for the darker environment.
3. **Camera Type**: We used a GoPro camera with a wide 16 mm lens, set to auto-focus.
4. **Video Settings**: We record in 4K resolution at 60 fps to maximize the number of keyframes extracted from the footage.


### Capture Tips

1. **Camera Motion**: Induce maximum parallax by moving around the objects you want to capture. This enhances depth information in the scene, which is critical for accurate 3D reconstruction.
2. **Capture Duration**: Capture a video for at least 5 minutes for each scene, ensuring extensive coverage of the environment.
3. **File Compression**: Due to video compression, interframes (interpolated frames) are included. Extract keyframes using FFMPEG to ensure the highest-quality images are used for reconstruction.
4. **Manual Check**: After extracting the keyframes, manually review and remove any images with excessive motion blur. This typically results in about 500 usable images per scene.
5. **Image Coverage**: Ensure that the images cover the scene extensively from different angles and viewpoints. The more diverse the captures, the better the resulting mesh and point cloud will be.

Make sure to also capture at least 3 images at known relative positions. They will allow to perform geo-registration in COLMAP afterwards, to align the z-axis with gravity and scale your frame of reference correctly. You can use a room corner or a table to do so.
### COLMAP

With your images in hand, we can use COLMAP to infer the camera poses and intrinsics.
```bash
cd third_party/2d_gaussian_splatting
python convert.py -s <path to your images folder>
```

Now you can perform geo-registration on your dataset.
First, create a text file `geo-registration.txt` like so:

```
image_name1.jpg X1 Y1 Z1
image_name2.jpg X2 Y2 Z2
image_name3.jpg X3 Y3 Z3
...
```

Then:
```bash
colmap model_aligner \
    --input_path ./sparse/0 \
    --output_path ./name_of_output_directory \
    --ref_images_path ./geo-registration.txt \
    --ref_is_gps 0 \
    --alignment_type custom \
    --alignment_max_error 3.0
```
where `sparse/0` is the path to your model directory. In the ideal case, there will be a single `distorted/sparse/0` directory and a single `sparse/0` output directory, in which case your model path is the latter. If there are several directories in `distorted/sparse`, pick the largest one and rename it to `0`, then re-run `python convert.py -s your/scene/path --skip_matching` to produce a new `sparse/0` output directory.

## Getting the Vision and Collision Mesh

Train the Gaussian Splatting model and render the mesh:

```bash
cd third_party/2d_gaussian_splatting
python train.py -s <path to COLMAP or NeRF Synthetic dataset>
python render.py -m <path to trained model> --skip-train --skip-test
```

If you encounter an incomplete mesh, you may need to adjust the `--sdf_trunc` or `--depth_trunc` parameters.

Next, decompose the mesh into convex subparts:

```bash
python coacd.py --mesh <path to your mesh>
```

You may need to tweak the `--preprocess_resolution` and `--threshold` parameters.

Afterwards, open the mesh in Blender, remove the ground and any artifacts, and save the processed mesh as `manual_postprocess.obj`.

Finally, generate a URDF to load the mesh in Pinocchio:

```bash
cd third_party/obj2urdf
python obj2urdf.py <path to the file.obj>
```

Copy `manual_postprocess.obj`, `manual_postprocess.urdf`, and `point_cloud.ply` to the `data` folder.

## Train and Test Your Navigation Policy

### Collect the Dataset

Collect some RGB and depth images to train the visual encoder:

```bash
python make_dataset.py
```

By default, the dataset will have 60,000 samples, but you can adjust this with the `--len_dataset` parameter.

### Train the Vision Encoder

Train the vision encoder:

```bash
python autoencoder.py
```

This script will also visualize the image, depth reconstruction, and depth ground truth at the end of training. Use the `--skip_train` argument to skip training and only view the visualizations. You can also adjust the batch size and number of epochs with the `--batch_size` and `--epochs` parameters.

![Depthreconstruction](vision_training.png)

### Train Your Navigation Policy

Now, you’re ready to train your navigation policy:

```bash
python train_policy.py
```

You can adjust the number of training steps with the `--training_steps` parameter.

### Test Your Navigation Policy

To test the navigation policy:

```bash
python test_nav_policy.py
```

This will open a window showing the agent’s behavior when instructed to go full throttle forward.

## Trying It Out on Upkie!

### In Simulation

1. In one terminal, start the Upkie simulation:

```bash
git clone https://github.com/upkie/upkie.git
cd upkie
git checkout 541b8ed686508c159a643f8c22316627a96f71ef
./start_simulation.sh
```

2. In another terminal, run the agent:

```bash
python run.py
```

This will open a window to visualize the FPV of the robot. With a joystick connected, your policy will correct your joystick inputs to avoid collisions.

### On Your Real Upkie

1. In one terminal, reset the Upkie system:

```bash
upkie_tool rezero
make run_pi3hat_spine
```

2. In another terminal, run the agent:

```bash
python run.py
```
