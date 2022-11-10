# ObjectReconstructor

Object shape reconstruction from RGB-D images. Auto-encoder trained on multi-view RGB-D images with pointcloud and voxel grid shape predictors.

## Install
### Prepare environment

Installation instructions were tested for Python 3.8, pytorch = 1.12.1/1.11.0, and cudatoolkit = 11.3.

```
git clone --recursive git@github.com:maturk/ObjectReconstructor.git
conda create -n obre python=3.8
conda activate obre
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch

pushd ./ObjectReconstructor
pip install -r requirements.txt
```

Follow [ngp_pl](https://github.com/kwea123/ngp_pl) installation instructions. The following method has been tested.
```
pushd ./ngp_pl
pip install torch-scatter==2.0.6
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

git clone https://github.com/NVIDIA/apex
pushd apex
pip install -v --disable-pip-version-check --global-option="--cuda_ext" --no-cache-dir ./ 
popd
pip install -r requirements.txt
popd
```

## Datasets
### Custom multi-view ShapeNet Dataset
Dataset of multi-view images of ShapeNet models with ground truth camera poses, depth maps, pointclouds, and voxel grids. Some examples of the synthetic renders can be seen below.

![cup](https://user-images.githubusercontent.com/30566358/201067906-197132c9-ccd9-470c-b0c4-65a0e439a30d.png)
![camera](https://user-images.githubusercontent.com/30566358/201069287-00936682-3635-4beb-a8b9-f5ff60b64f9a.png)
![phone](https://user-images.githubusercontent.com/30566358/201069584-6e05a430-4e6b-4b90-9023-bc72f43d0b93.png)
![pepsi](https://user-images.githubusercontent.com/30566358/201070290-3d8b2e27-f89a-4943-9946-b0277bb831d7.png)
![laptop](https://user-images.githubusercontent.com/30566358/201070676-77d6207b-54a8-4c3c-8ec4-2aecd91b8657.png)
![guitar](https://user-images.githubusercontent.com/30566358/201070941-a2aad8e1-bcad-44ae-865d-78dc2dd182be.png)

<!--![bowl](https://user-images.githubusercontent.com/30566358/201068558-f08f935a-89a4-4495-a258-b1bbd2d08f15.png)-->
<!-- ![bottle](https://user-images.githubusercontent.com/30566358/201069912-eb07889a-4444-43e7-a131-fc803598c320.png) -->


### Prepare Blender Dataset
Tested on Blender version 2.93.9. Note, Blender version <3.0 required. Python API changes in 3.0+ do not work with the preprocess script. To create the dataset of rgb, depth, and ground truth pointclouds and voxel grids, open Blender and run preprocess_blender.py script. 

## Results
### Voxel results
Ground truth vs predicted:

Work in progress...
