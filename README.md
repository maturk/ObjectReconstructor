# ObjectReconstructor

Object shape reconstruction from synthetic RGB-D images. Auto-encoder trained on multi-view RGB-D images with pointcloud and voxel grid shape predictors.

## Features
* [x] Multi-view/single-view encoder support for both training and inference time.
* [x] Supports both RGB-D channel or Depth-only input.
* [x] Supports point cloud or voxel decoders for shape reconstruction.

## Installation Instructions
### Prepare Environment

Installation instructions were tested for Python 3.8, pytorch = 1.12.1/1.11.0, and cudatoolkit = 11.3.

```
git clone --recurse-submodules git@github.com:maturk/ObjectReconstructor.git
conda create -n obre python=3.8
conda activate obre
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch

cd ./ObjectReconstructor
pip install -r requirements.txt
```

## Datasets
### Multi-view ShapeNet Dataset
Dataset of multi-view images of ShapeNet models with ground truth camera poses, depth maps, pointclouds, and voxel grids. Some examples of the synthetic renders can be seen below.

![cup](https://user-images.githubusercontent.com/30566358/201067906-197132c9-ccd9-470c-b0c4-65a0e439a30d.png)
![camera](https://user-images.githubusercontent.com/30566358/201069287-00936682-3635-4beb-a8b9-f5ff60b64f9a.png)
![phone](https://user-images.githubusercontent.com/30566358/201069584-6e05a430-4e6b-4b90-9023-bc72f43d0b93.png)
![pepsi](https://user-images.githubusercontent.com/30566358/201070290-3d8b2e27-f89a-4943-9946-b0277bb831d7.png)
![laptop](https://user-images.githubusercontent.com/30566358/201070676-77d6207b-54a8-4c3c-8ec4-2aecd91b8657.png)
![guitar](https://user-images.githubusercontent.com/30566358/201070941-a2aad8e1-bcad-44ae-865d-78dc2dd182be.png)

<!--![bowl](https://user-images.githubusercontent.com/30566358/201068558-f08f935a-89a4-4495-a258-b1bbd2d08f15.png)-->
<!-- ![bottle](https://user-images.githubusercontent.com/30566358/201069912-eb07889a-4444-43e7-a131-fc803598c320.png) -->

### Download Training Dataset
You can download a premade train/test split by running the following script:
```
bash ./download_custom_dataset.sh 
```

### Create Your Own Dataset with Blender
Tested on Blender version 2.93.9. Note, Blender version <3.0 required. To create your own custom dataset of rgb, depth, and ground truth point clouds and voxel grids, open Blender, modify the download directory of your shapenet model/texture files, and run preprocess_blender.py script.

## Training Scripts
### RGB-D fusion vs depth-only training
I provide training scripts to train auto-encoders for both RGB-D or Depth only channels. RGB-D training works with a modified [Dense-Fusion](https://github.com/j96w/DenseFusion) architecture that uses both color and depth channels to create an embedding vector. Multiple-views of the same object are fused into one embedding. Depth only training works with a PointNet encoder. Point cloud and voxel grid decoders are simple MLPs.

## Results
### Voxel Depth Only Autoencoder
The following results were obtained with the depth only voxel grid autoencoder as shown in the architecure diagram:

<img width="600" alt="table_res"  src ="https://user-images.githubusercontent.com/30566358/214982861-29b94212-badf-487f-8b47-334877a2c83e.png" >
<img width="600" alt="table_res" src="https://user-images.githubusercontent.com/30566358/214982418-c3c95ee1-7103-4380-ab95-769a1f096714.png">

For more qualitative results see: https://maturk.github.io/projects/object_reconstruction.html









