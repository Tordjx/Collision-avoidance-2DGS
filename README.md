## How to reproduce the results ?

Create the conda environment
Creating a GS dataset
Using 2DGS
Getting the collision mesh
Train your navigation policy
TODO : make a nice env file to make it work nice and quick
## Installation

### Conda environment

Create the `vision_agent` environment:

```console
conda env create -f environment.yaml
```

Pay attention to the version of PyTorch resolved when the environment is created. It should contain a version of CUDA:

```diff
- bad
-   + pytorch                       2.4.1  cpu_mkl_py311hb499fb8_100  conda-forge       36MB
+ good
+   + pytorch                       2.4.1  cuda118_py313h49748f1_302  conda-forge       26MB
```

Activate the fresh environment:

```console
conda activate vision_agent
```

### Install 2D Gaussian splatting

Install the 2DGS rasterizer:

```consoleUsing 2DGS

```console
pip install ./third_party/2d-gaussian-splatting/submodules/simple-knn
```

Check out troubleshooting below if you get an error.

Finally, apply the following patch to the Gaussian renderer in `third_party/2d-gaussian-splatting`:

```diff
--- a/gaussian_renderer/__init__.py
+++ b/gaussian_renderer/__init__.py
@@ -75,8 +75,9 @@ def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor,
         cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
     else:
         scales = pc.get_scaling
+        scales = scales[:, :2]
         rotations = pc.get_rotation
```

## Troubleshooting

### Error: namespace "std" has no member "uintptr\_t"

Add the following line to `third_party/2d-gaussian-splatting/submodules/diff-surfel-rasterization/cuda_rasterizer/rasterizer_impl.h`:

```cpp
#include <cstdint>
```

### Error: undefined FLT\_MAX constant

Add the following line ([source](https://github.com/pytorch/audio/pull/3811/files)) to `third_party/2d-gaussian-splatting/submodules/simple-knn/simple_knn.cu`:

```cpp
#include <float.h>
```
## Creating a GS dataset

Camera settings
Capture tips
COLMAP

## Getting the vision and the collision mesh

python train.py -s bidule

python render.py -m truc --skip-train --skip-test 

you might need to tweak sdf trunc (verifier) if you find yourself with an incomplete mesh

python coacd.py --mesh meshpath --relevant_coacd_params truc

expliquer les deux parametres importatnts de coacd

blender

## Train and test your navigation policy 

# Collect the dataset to train the vision encoder
python make_dataset.py --len_dataset N 
the default is 60000
# Train the vision encoder
python train_encoder.py --batch_size 256 --epochs 10000

python train_policy.py 

python test_nav_policy.py to vizualize in the pure navigation env

### Trying it out on Upkie !

# In sim

in upkie
./start_simulation.sh

in another terminal
python run.py 

# On your real Upkie

upkie_tool rezero

make run_pi3hat_spine

python run.py