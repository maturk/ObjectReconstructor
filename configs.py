'''
    Global configs
'''

import numpy as np

# Blender configs
DEPTH_SCALE= 0.25

# Voxel configs
CUBIC_SIZE = 2
VOXEL_RESOLUTION = 128
VOXEL_SIZE = V = CUBIC_SIZE/ VOXEL_RESOLUTION
VOXEL_OCCUPANCY_TRESHOLD = 0.6
GRID_MIN = np.array([-1,-1,-1])
GRID_MAX = np.array([1,1,1])

# PC configs
PC_NUM_POINTS = 1028
PC_GT_NUM_POINTS = 2048