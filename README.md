# ObjectReconstructor

Object shape reconstruction from RGB-D images. Object centric shape and appearance code auto-encoder trained on synthetic ShapeNet models with Blender.

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

### Prepare Blender Dataset
Tested on Blender version 2.93.9. Note, Blender version <3.0 required. Python API changes in 3.0+ do not work with the preprocess script.   To create the dataset of rgb, depth, and ground truth point clouds, open Blender and run preprocess_blender.py script. 

Work in progress...
