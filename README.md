## How to reproduce the results ?

Create the conda environment
Creating a GS dataset
Using 2DGS
Getting the collision mesh
Train your navigation policy
TODO : make a nice env file to make it work nice and quick
## Installation

First clone the repository recursively 
git clone ... --recursive

### Conda environment

Add the following line ([source](https://github.com/pytorch/audio/pull/3811/files)) to `third_party/2d-gaussian-splatting/submodules/simple-knn/simple_knn.cu`:

```cpp
#include <float.h>
```

Create the `vision_agent` environment:

```console
conda env create -f environment.yaml

conda activate vision_agent
```


## Creating a GS dataset

Camera settings
Capture tips
COLMAP

## Getting the vision and the collision mesh

Train the gaussian splatting model and render the mesh

```console
cd third_party/2d_gaussian_splatting

python train.py -s <path to COLMAP or NeRF Synthetic dataset> 

python render.py -m <path to trained model> --skip-train --skip-test 
```
you might need to tweak --sdf_trunc or --depth_trunc if you find yourself with an incomplete mesh

Next, decompose the mesh in convex subparts using this scrip

```console
python coacd.py --mesh <path to your mesh> 
```
You may tweak --preprocess_resolution and --threshold parameters.

Next, open this mesh in blender and remove the ground, and any artifacts that there might be
Save and name your postprocessed mesh manual_postprocess.obj

Finally, generate a urdf to be able to load it in pinocchio

```console
cd third_party/obj2urdf
python obj2urdf.py <path to the file.obj>
```

Copy manual_postprocess.obj, manual_postprocess.urdf, and point_cloud.ply to the folder data.
## Train and test your navigation policy 

# Collect the dataset to train the vision encoder
Collect some RGB and depth image to learn the visual implicit representation

```console
python make_dataset.py 
```

The default length of the dataset is 60000, but you may adjust it using the --len_dataset parameter.

You may now train the visual encoder.
```console
python autoencoder.py 
```
This script will also generate a vizualization of the image, depth reconstruction, and depth ground truth at the end of the training.
You may skip the training to only get the vizualization using the --skip_train argument.
You may also use the --batch_size and --epochs parameters.

You are now ready to train your navigation policy.

```console
python train_policy.py 

```
You may adjust the number of training steps with the --training_steps parameter.

You can try out your navigation policy.
```console

python test_nav_policy.py
```
This script will open a window for you to see your agent behave when told to go full throttle forward.

### Trying it out on Upkie !

You are now ready to test your navigation policy in a simulator and/or on your real Upkie.
# In sim

In one terminal 
```console
git clone https://github.com/upkie/upkie.git
cd upkie
./start_simulation.sh
```
In another terminal
```console
python run.py 
```
Note that this script will also open a window for you to visualize the FPV of the robot.

# On your real Upkie

In one terminal :
```console 
upkie_tool rezero
make run_pi3hat_spine
```

In another :
```console
python run.py
```

