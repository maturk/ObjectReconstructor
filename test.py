
from cmath import atanh
from re import A, I
from turtle import width
import torch
import pickle
import os
import torch.nn as nn
from numpy.random import randint
from PIL import ImageSequence
import cv2
import sys
import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt
import open3d as o3d

class NOCSDataset(torch.utils.data.Dataset):
    def __init__(self, split = 'val', rootDirectory = '/Users/maturk/data/'):
        self.split = split
        assert self.split in ['real_test', 'val']
        self.rootDirectory = rootDirectory
        if (self.split == 'val') :
            self.dataRoot = os.path.join(rootDirectory, 'val')
            self.gtRoot = os.path.join(rootDirectory, 'gts/val')
            names = self.get_image_names(self.dataRoot)
        self.color_paths = names['color_path']
        self.depth_paths = names['depth_path']
        self.coord_paths = names['coord_path']
        self.mask_paths = names['mask_path']

    def __getitem__(self, folder_idx, idx):
        pass

    def __len__(self):
        return (len(self.color_paths))
    
    def _get_random(self):
        ind = randint(0, high = self.__len__())
        print('len ', self.__len__())
        image = cv2.imread(self.color_paths[ind])
        depth = cv2.imread(self.depth_paths[ind])
        coord = cv2.imread(self.coord_paths[ind])
        mask  = cv2.imread(self.mask_paths[ind])
        
        return {
           'image' : image,
           'depth' : depth,
           'coord' : coord,
           'mask' : mask,
           'image_number' : ind
        }

    def get_image_names(self, dataRoot):
        folders = os.listdir(dataRoot)
        folders = sorted(folders)
        image_ids = range(len(folders)-1)
        color_paths = []
        depth_paths = []
        coord_paths = []
        mask_paths = []
        image_numbers = []
        image_number= 0
        for folder in folders:
            folder = os.path.join(dataRoot, folder)
            image_paths = glob.glob(os.path.join(folder, '*_color.png'))
            image_paths = sorted(image_paths)
            for color in image_paths:
                image_id = os.path.basename(color).split('_')[0]
                image_path = os.path.join(folder,image_id)
                meta_path = image_path + '_meta.txt'
                inst_dict = {}
                with open(meta_path, 'r') as f:
                    for line in f:
                        line_info = line.split(' ')
                        inst_id = int(line_info[0])
                        cls_id = int(line_info[1])
                        inst_dict[inst_id] = cls_id
            
                color_path = image_path + '_color.png'
                color_paths.append(color_path)
                coord_path = image_path + '_coord.png'
                coord_paths.append(coord_path)
                depth_path = image_path + '_depth.png'
                depth_paths.append(depth_path)
                mask_path = image_path + '_mask.png'
                mask_paths.append(mask_path)
                image_numbers.append(image_number)

        image_number+=1

        return {
            'color_path' : color_paths,
            'depth_path' : depth_paths,
            'coord_path' : coord_paths,
            'mask_path' : mask_paths,
            'image_number' : image_numbers
        }
                

device = torch.device("mps")

class PointCloudEncoder(nn.Module):
    def __init__(self, emb_dim):
        super(PointCloudEncoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(256, 256, 1)
        self.conv4 = nn.Conv1d(256, 1024, 1)
        self.fc = nn.Linear(1024, emb_dim)

    def forward(self, xyz): 
        """
        Args:
            xyz: (B, 3, N)
        """
        np = xyz.size()[2]
        x = F.relu(self.conv1(xyz))
        x = F.relu(self.conv2(x))
        global_feat = F.adaptive_max_pool1d(x, 1)
        x = torch.cat((x, global_feat.repeat(1, 1, np)), dim=1)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.squeeze(F.adaptive_max_pool1d(x, 1), dim=2)
        embedding = self.fc(x)
        return embedding


if __name__ == "__main__":
    
    depth = np.array(cv2.imread('/Users/maturk/data/test/02876657/dacc6638cd62d82f42ebc0504c999b/15_texture1_depth_depth0001.png')[:,:,0])
    print((depth))
    mask = depth<255
    depth_masked = depth[mask][:, np.newaxis]
    print(np.shape(depth_masked))
    plt.imshow(depth)
    plt.show()

    xmap = np.array([[j for i in range(640)] for j in range(480)])
    xmap = xmap[mask][:,np.newaxis]
    ymap = np.array([[i for i in range(640)] for j in range(480)])
    ymap = ymap[mask][:,np.newaxis]
    cam_scale = 250
    cam_fx = 50 *10
    cam_fy = 50 *10
    cam_cx = 240 #240
    cam_cy = 320 #320
    z = depth_masked / cam_scale
    x = (ymap - cam_cx) * z / cam_fx
    y = (xmap - cam_cy) * z / cam_fy
    cloud = np.concatenate((x, y, z), axis=1)
    print(cloud)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    o3d.visualization.draw_geometries([pcd])

    #dataSet = NOCSDataset(rootDirectory='/Users/maturk/data/', split = 'val')
    #random = dataSet._get_random()
    #image = random['image']
    #num = random['image_number']
    #cv2.imshow('image', image)
    #cv2.waitKey(0) 