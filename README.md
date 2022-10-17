# ObjectReconstructor

Object shape reconstruction from RGB-D images. Object centric shape and appearance code auto-encoder trained on synthetic ShapeNet models with Blender.

## Install
### Prepare environment

Installation instructions were tested for Python 3.8, pytorch = 1.12.1, and cudatoolkit = 11.3.

```
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
conda env create -f environment.yml
```

### Prepare Blender Dataset
Tested on Blender version 2.93.9. Note, Blender version <3.0 required. Python API changes in 3.0+ do not work with the preprocess script.   To create the dataset of rgb, depth, and ground truth point clouds, open Blender and run preprocess_blender.py script. 

Work in progress...
