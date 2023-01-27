''' 
Simple script to make gifs out of result files
'''

import imageio
import argparse
import glob
import os
from natsort import natsorted

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', type=str, default=f'{os.getcwd()}/inference/results/voxels', help='directory to save inference results')

NAME = 'display'
if __name__ == "__main__":
    opt = parser.parse_args() 
    filenames =  natsorted(glob.glob(os.path.join(opt.result_dir, 'images', 'point_clouds',f'pred_*{NAME}*')))
    print(filenames)
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    kargs = { 'duration': 0.5 }
    imageio.mimsave(os.path.join(os.getcwd(),'scratch', f'{NAME}_gif.gif'), images, **kargs)