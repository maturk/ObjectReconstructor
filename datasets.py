import torch
import os
from numpy.random import randint
import cv2
import glob
import numpy as np
import open3d as o3d
import json
from scipy.spatial.transform import Rotation 
from utils import pc_local_to_pc_global, pc_to_dmap, blender_dataset_to_ngp_pl
from configs import *


class BlenderDataset(torch.utils.data.Dataset):
    """ Blender dataset loader

    """
    def __init__(self, mode = 'train', save_directory  = '/Users/maturk/data/test', num_points = 1024, voxel_size = 128, num_views = 15):
        self.objects = []
        self.root_directory = save_directory
        self.mode = mode
        self.get_object_paths()
        self.num_points = num_points
        self.voxel_size = voxel_size
        self.num_views = num_views
        
    def __getitem__(self, index):
        object = self.objects[index]
        K, cam_scale, cam_fx, cam_fy, cam_cx, cam_cy = self.get_intrinsics(return_params=True)
        gt_pc = np.asarray(o3d.io.read_point_cloud(object['gt_pc']).points, dtype = np.float32).transpose()
        gt_grid = np.load(object['gt_grid'])
        with open(object['poses']) as f:
            data = f.read()
        data = data.split(';')
        data.pop(-1) # remove empty entry
        poses = []
        for i in range(len(data)):
            pose = np.eye(4)
            js = json.loads(data[i])
            x = js['x']
            y = js['y']
            z = js['z']
            eul_x = js['eul_x']
            eul_y = js['eul_y']
            eul_z = js['eul_z']
            rotation = Rotation.from_euler('xyz', [eul_x, eul_y, eul_z], degrees=False).as_matrix()
            position = np.array([x,y,z])
            pose[:3,3] = position
            pose[:3,:3] = rotation
            poses.append(pose)
        depths = []
        colors = []
        masks = []
        for i, (dmap, color) in enumerate(zip(object['depth_paths'][0:self.num_views], object['color_paths'][0:self.num_views])):
            try:
                dmap_img = np.array(cv2.imread(dmap)[:,:,0], dtype=np.float32)
                color_img = np.array(cv2.imread(color), dtype=np.float32)
                mask = dmap_img<255
                w = np.shape(color_img)[1]
                h = np.shape(color_img)[0]
                dmap_masked = dmap_img[mask][:, np.newaxis]
                xmap = np.array([[j for i in range(h)] for j in range(w)])
                xmap = xmap[mask][:,np.newaxis]
                ymap = np.array([[i for i in range(h)] for j in range(w)])
                ymap = ymap[mask][:,np.newaxis]
                z = - (1/DEPTH_SCALE) * dmap_masked / cam_scale
                x = (ymap - cam_cx) * z / cam_fx
                y = (xmap - cam_cy) * z / cam_fy
                cloud = np.concatenate((x, y, z), axis=1)
                cloud = pc_local_to_pc_global(cloud, K, poses[i], blender_pre_rotation=True)
                cloud, idxs = self.pc_down_sample(cloud, self.num_points)
                mask = idxs
                depths.append(cloud)
                colors.append(color_img)
                masks.append(mask)
            except:
                print('FAILED, dataset paths corrupted')
                print(object['class_dir'], object['object_dir'])
                return self.__getitem__(np.random.randint(0, high = self.__len__() -1))
        
        return {
            'colors' : torch.tensor(np.array(colors), dtype=torch.float32),
            'depths' : torch.tensor(np.array(depths), dtype=torch.float32),
            'gt_pc'  : torch.tensor(gt_pc, dtype=torch.float32),
            'gt_grid': torch.tensor(gt_grid, dtype=torch.float32),
            'masks'   : torch.tensor(np.array(masks), dtype=torch.float32),
            'class_dir' : object['class_dir'],
            'object_dir' : object['object_dir'],
            'poses' : torch.tensor(np.array(poses[0:self.num_views]), dtype=torch.float32)
        }
        
    def __len__(self):
        return len((self.objects))
    
    def get_intrinsics(self, h = 240, w = 240, return_params = False):
        cam_scale = 250
        cam_fx = 333
        cam_fy = 333
        cam_cx = int(h/2) 
        cam_cy = int(w/2)
        K = torch.zeros((3,4), dtype= torch.float)
        K[0,0] = cam_fx
        K[1,1] = cam_fy
        K[2,2] = 1
        K[0,2] = cam_cx
        K[1,2] = cam_cy
        if return_params:
            return K, cam_scale, cam_fx, cam_fy, cam_cx, cam_cy
        return K 
        
    def get_object_paths(self):
        folders = os.listdir(self.root_directory)
        folders = sorted(folders)
        object_classes = range(len(folders)-1)
        object_number= 0
        for class_dir in folders:
            if not class_dir.startswith('.'):
                folder = os.path.join(self.root_directory, class_dir)
                for object_dir in sorted(os.listdir(folder)):
                    if not object_dir.startswith('.'):
                        image_paths = glob.glob(os.path.join(folder, object_dir, '*color.png'))
                        image_paths = sorted(image_paths)
                        dmap_paths = glob.glob(os.path.join(folder, object_dir, '*depth*.png'))
                        dmap_paths = sorted(dmap_paths)
                        color_paths = [None] * len(image_paths)
                        depth_paths = [None] * len(dmap_paths)
                        for image, dmap in zip(image_paths, dmap_paths):
                            image_id = int(os.path.basename(image).split('_')[0])
                            color_paths[image_id] = image 
                            dmap_id = int(os.path.basename(dmap).split('_')[0])
                            depth_paths[dmap_id] = dmap
                        if self.mode == 'train':
                            gt_pc = glob.glob(os.path.join(folder, object_dir, 'pc.pcd'))[0]
                            gt_grid = glob.glob(os.path.join(folder, object_dir, 'grid.npy'))[0]
                            poses = glob.glob(os.path.join(folder, object_dir, 'poses.txt'))[0]
                        else:
                            gt_pc = None
                            gt_grid = None
                            poses = None
                        
                        self.add_object(object_number, color_paths, depth_paths, class_dir, object_dir, gt_pc, gt_grid, poses=poses)
                        object_number+=1

    def add_object(self, object_number, color_paths, depth_paths, class_dir, object_dir, gt_pc = None, gt_grid = None, poses = None):
        object_info = { 'object_number' : object_number,
            'color_paths' : color_paths,
            'depth_paths' : depth_paths,
            'class_dir' : int(class_dir),
            'object_dir' : object_dir,
            'gt_pc'     : gt_pc,
            'gt_grid'   : gt_grid,
            'poses' : poses
        }
        self.objects.append(object_info)

    def pc_down_sample(self, pc, num_points):
        xyz = pc
        num_xyz = pc.shape[0]
        if num_xyz >= self.num_points:
            idxs = np.random.choice(num_xyz, num_points)
            xyz = xyz[idxs, :]
        else:
            # if not enough points to downsample, randomly repeat points in input PC
            rem = num_points - num_xyz
            rem_idx = np.random.choice(num_xyz, rem)
            xyz_rem = xyz[rem_idx, :]
            xyz =  np.concatenate((xyz, xyz_rem), axis=0)
            idxs = np.random.choice(num_xyz, num_points)
            xyz = xyz[idxs, :]
        return xyz, idxs


class NOCSDataset(torch.utils.data.Dataset):
    """NOCS Dataset loader
    
    """
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
                                

if __name__ == "__main__":
    dataset = BlenderDataset(save_directory  = '/home/maturk/data/small_set')
    pass
    