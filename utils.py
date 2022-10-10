import torch
import os
import torch.nn as nn
from numpy.random import randint
import cv2
import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt
import open3d as o3d
import torch.nn.functional as F


class BlenderDataset(torch.utils.data.Dataset):
    def __init__(self, mode = 'train', save_directory  = '/Users/maturk/data/test', num_points = 1024):
        self.objects = []
        self.root_directory = save_directory
        self.mode = mode
        self.get_object_paths()
        self.num_points = num_points
        
        
    def __getitem__(self, index):
        object = self.objects[index]
        depths = []
        colors = []
        masks = []
        for dmap, color in zip(object['depth_paths'][0:15], object['color_paths'][0:15]):
            try:
                dmap_img = np.array(cv2.imread(dmap)[:,:,0], dtype=np.float32)
                color_img = np.array(cv2.imread(color), dtype=np.float32)
                mask = dmap_img<255
                dmap_masked = dmap_img[mask][:, np.newaxis]
                xmap = np.array([[j for i in range(640)] for j in range(480)])
                xmap = xmap[mask][:,np.newaxis]
                ymap = np.array([[i for i in range(640)] for j in range(480)])
                ymap = ymap[mask][:,np.newaxis]
                cam_scale = 250
                cam_fx = 50 *10
                cam_fy = 50 *10
                cam_cx = 240 #240
                cam_cy = 320 #320
                z = dmap_masked / cam_scale
                x = (ymap - cam_cx) * z / cam_fx
                y = (xmap - cam_cy) * z / cam_fy
                cloud = np.concatenate((x, y, z), axis=1)
                cloud, idxs = self.pc_down_sample(cloud, self.num_points)
                mask = idxs
                depths.append(cloud)
                colors.append(color_img)
                masks.append(mask)
            except:
                print('FAILED, dataset corrupted')
                print('Object directory: ', object['object_dir'])
                print('Class directory: ', object['class_dir'])
            
        gt_pc = np.asarray(o3d.io.read_point_cloud(object['gt_pc']).points, dtype = np.float32).transpose()
        return {
            'colors' : torch.tensor(np.array(colors)),
            'depths' : torch.tensor(np.array(depths)),
            'gt_pc'  : torch.tensor(gt_pc),
            'masks'   : torch.tensor(np.array(masks)),
            'class_dir' : object['class_dir'],
            'object_dir' : object['object_dir']
        }
        
    def __len__(self):
        return len((self.objects))
    
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
                            gt_pc = glob.glob(os.path.join(folder, object_dir, 'pc.ply'))[0]
                        else:
                            gt_pc = None
                        self.add_object(object_number, color_paths, depth_paths, class_dir, object_dir, gt_pc)
                        object_number+=1

    def add_object(self, object_number, color_paths, depth_paths, class_dir, object_dir, gt_pc):
        object_info = { 'object_number' : object_number,
            'color_paths' : color_paths,
            'depth_paths' : depth_paths,
            'class_dir' : int(class_dir),
            'object_dir' : object_dir,
            'gt_pc'     : gt_pc
        }
        self.objects.append(object_info)

    def pc_down_sample(self, pc, num_points):
        xyz = pc
        num_xyz = pc.shape[0]
        assert num_xyz >= self.num_points, 'Not enough points in shape.'
        idxs = np.random.choice(num_xyz, self.num_points)
        xyz = xyz[idxs, :]
        # To do : data augmentation and random noise
        return xyz, idxs
    


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
                

def shapenet_pc_sample(shapenet_directory = '/home/asl-student/mturkulainen/data/Shapenet_small', save_directory = '/home/asl-student/mturkulainen/data/test', sample_points = 2048):
    counter = 0
    for folder in sorted(os.listdir(shapenet_directory))[0:1]:
        if not folder.startswith('.'):
            for object_dir in (os.listdir(os.path.join(shapenet_directory, folder))):
                model_path = os.path.join(shapenet_directory, folder, object_dir, 'models')
                models = glob.glob(os.path.join(model_path, '*.obj'))
                if models != []:
                    print(object_dir)
                    model = models[0]
                    mesh = o3d.io.read_triangle_mesh(model)
                    points = mesh.sample_points_uniformly(sample_points)
                    o3d.io.write_point_cloud(os.path.join(save_directory, folder, object_dir, 'pc.ply'), points, compressed = False)
                    counter += 1
                else:
                    continue

# Faulty files:
# abe557fa1b9d59489c81f0389df0e98a
# 194f4eb1707aaf674c8b72e8da0e65c5
# 5979870763de5ced4c8b72e8da0e65c5

if __name__ == "__main__":
    dataset = BlenderDataset(save_directory  = '/home/maturk/data/test')
    #dataset.get_object_paths()
    objects = []

    [objects.append(dataset.__getitem__(idx)) for idx in range(0,5)]
    from pytorch_metric_learning import losses
    contrastive_loss = losses.contrastive_loss.ContrastiveLoss()

    tens = torch.rand(1,3)
    tens2 = torch.rand(1,3)
    print(tens,tens2)
    embs = []
    for i in range(0,5):
        embs.append(tens)
    #embs.append(tens2)
    embs = torch.cat(embs, dim = 0)
    print(embs.shape)
    labels = [torch.tensor(int(objects[i]['class_dir'])) for i in range(0,5)]
    labels = torch.tensor(labels)
    print(labels)
    loss = contrastive_loss(embs,labels)
    print(loss)
    #pc_gt = object['gt_pc']
    #print(np.shape(pc_gt))
    #pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(pc_gt.transpose())
    #o3d.visualization.draw_geometries([pcd])
    #shapenet_pc_sample()
