from turtle import color, width
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
import json
from scipy.spatial.transform import Rotation 

DEPTH_SCALE= 0.25

class BlenderDataset(torch.utils.data.Dataset):
    """ Blender dataset loader

    """
    def __init__(self, mode = 'train', save_directory  = '/Users/maturk/data/test', num_points = 1024, num_views = 15):
        self.objects = []
        self.root_directory = save_directory
        self.mode = mode
        self.get_object_paths()
        self.num_points = num_points
        self.num_views = num_views
        
    def __getitem__(self, index):
        object = self.objects[index]
        K, cam_scale, cam_fx, cam_fy, cam_cx, cam_cy = self.get_intrinsics(return_params=True)
        gt_pc = np.asarray(o3d.io.read_point_cloud(object['gt_pc']).points, dtype = np.float32).transpose()
        
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
                return self.__getitem__(np.random.randint(0, high = self.__len__() -1))
        
        return {
            'colors' : torch.tensor(np.array(colors), dtype=torch.float32),
            'depths' : torch.tensor(np.array(depths), dtype=torch.float32),
            'gt_pc'  : torch.tensor(gt_pc, dtype=torch.float32),
            'masks'   : torch.tensor(np.array(masks), dtype=torch.float32),
            'class_dir' : object['class_dir'],
            'object_dir' : object['object_dir'],
            'poses' : torch.tensor(np.array(poses), dtype=torch.float32)
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
                            poses = glob.glob(os.path.join(folder, object_dir, 'poses.txt'))[0]
                        else:
                            gt_pc = None
                        self.add_object(object_number, color_paths, depth_paths, class_dir, object_dir, gt_pc, poses=poses)
                        object_number+=1

    def add_object(self, object_number, color_paths, depth_paths, class_dir, object_dir, gt_pc = None, poses = None):
        object_info = { 'object_number' : object_number,
            'color_paths' : color_paths,
            'depth_paths' : depth_paths,
            'class_dir' : int(class_dir),
            'object_dir' : object_dir,
            'gt_pc'     : gt_pc,
            'poses' : poses
        }
        self.objects.append(object_info)

    def pc_down_sample(self, pc, num_points):
        xyz = pc
        num_xyz = pc.shape[0]
        assert num_xyz >= self.num_points, 'Not enough points in shape.'
        idxs = np.random.choice(num_xyz, num_points)
        xyz = xyz[idxs, :]
        # To do : data augmentation and random noise
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
        
def pc_local_to_pc_global(pc, K, pose, blender_pre_rotation = True):
    """ Transform local point cloud into global frame

    Args:
        pc (pc): local pc
        K (array): [4,4] intrinsics
        pose (array): [4,4] pose
        blender_pre_rotation (bool, optional): If using Blender ShapeNet dataset, objects are pre-rotated by 90 degrees about x-axis. Defaults to True.
    """
    K = np.array(K)
    pose  = np.array(pose)
    if np.shape(pc)[1] !=3:
        pc = np.transpose(pc)
    if blender_pre_rotation:
        r = Rotation.from_euler('x', -90, degrees=True)
        pre_rot_T = np.eye(4)
        pre_rotation = r.as_matrix()
        pre_rot_T[:3,:3] = pre_rotation
    else: pre_rot_T = np.eye(4)
    
    pc_out_global = np.ndarray((pc.shape[0], pc.shape[1]))
    for i in range(pc.shape[0]):
        p = np.asarray(pc[i])
        point = np.ones((4,1))
        point[:3,0]= p
        temp = (pose @ point)[0:3]
        temp = pre_rot_T[:3,:3] @ temp
        pc_out_global[i][0]= temp[0]
        pc_out_global[i][1]= temp[1]
        pc_out_global[i][2]= temp[2]
        
    return pc_out_global
                
def pc_to_dmap(pc, K, pose, blender_pre_rotation = True):
    """ Point cloud in global frame to depth map in pose frame conversion

    Args:
        pc (array): [n,3]
        K (intrinsics): [4,4]
        pose (pose): [4,4]
        blender_pre_rotation (bool, optional): If using Blender ShapeNet dataset, objects are pre-rotated by 90 degrees about x-axis. Defaults to True.

    Returns:
        image: [H,W] depth map image defined by K
    """
    K = np.array(K)
    if np.shape(pc)[1] !=3:
        pc = np.transpose(pc)
    if blender_pre_rotation:
        r = Rotation.from_euler('x', 90, degrees=True)
        pre_rot_T = np.eye(4)
        pre_rotation = r.as_matrix()
        pre_rot_T[:3,:3] = pre_rotation
    pose_inv = torch.zeros((4,4))
    r_inv = torch.Tensor(np.transpose(pose[0:3,0:3]))
    p_inv = -1 * r_inv @ pose[0:3,3]
    pose_inv [0:3,0:3] = r_inv
    pose_inv [0:3,3] = p_inv
    dmap = []
    for i in range(pc.shape[0]):
        p = np.asarray(pc[i])
        point = np.ones((4,1))
        point[:3,0]= p
        if blender_pre_rotation:
            point = pre_rot_T@point
        p_cam = (pose_inv @ point).float()
        p_out= torch.Tensor(K) @ p_cam
        p_out = p_out/p_out[2]
        p_out = torch.transpose(p_out, 0, 1)
        dmap.append(p_out)
    dmap = torch.cat(dmap, dim = 0)
    indices = torch.Tensor.numpy(dmap)
    img = indices[:,0:2]
    image = np.ones((int(K[0,2]*2),int(K[1,2]*2),3))
    for i in range(np.shape(img)[0]):
        image[int(img[i,1]), int(img[i,0]), :] = [0,0,0] 
    return image
 
def shapenet_pc_sample(shapenet_directory = '/home/maturk/data/Shapenet_small', save_directory = '/home/maturk/data/test2', sample_points = 2048):
    """ ShapeNet point cloud generator

    Args:
        shapenet_directory (str, optional): ShapeNet model directory (obj and texture files). Defaults to '/home/maturk/data/Shapenet_small'.
        save_directory (str, optional): Save directory. Defaults to '/home/maturk/data/test2'.
        sample_points (int, optional): Number of points to sample for point cloud. Defaults to 2048.
    TODO:     # Faulty files: # abe557fa1b9d59489c81f0389df0e98a # 194f4eb1707aaf674c8b72e8da0e65c5 # 5979870763de5ced4c8b72e8da0e65c5 # b838c5bc5241a44bf2f2371022475a36 # c50c72eefe225b51cb2a965e75be701c # 9d453384794bc58b9a06a7de97b096dc # 1a0a2715462499fbf9029695a3277412
    """
    counter = 0
    for folder in sorted(os.listdir(shapenet_directory)):
        if not folder.startswith('.'):
            for object_dir in (os.listdir(os.path.join(shapenet_directory, folder))):
                files_in_save_folder = glob.glob(os.path.join(save_directory, folder, object_dir))
                if os.path.join(save_directory, folder, object_dir) in files_in_save_folder:
                    model_path = os.path.join(shapenet_directory, folder, object_dir, 'models')
                    models = glob.glob(os.path.join(model_path, '*.obj'))
                    if models != []:
                        model = models[0]
                        mesh = o3d.io.read_triangle_mesh(model)
                        points = mesh.sample_points_uniformly(sample_points)
                        o3d.io.write_point_cloud(os.path.join(save_directory, folder, object_dir, 'pc.pcd'), points, compressed = False)
                        counter += 1
                    else:
                        continue


def blender_dataset_to_ngp_pl(save_directory):
    ''' blender_dataset_to_ngp_pl
    
    Args:
        save_directory (str, optional): Save directory. Defaults to '/home/maturk/data/test2'.
    #TODO: add custom K [4x4]:
    '''
    import os
    import shutil
    rgb_path = os.path.join(save_directory, "rgb")
    pose_path = os.path.join(save_directory, "pose")
    intrinsics_path = os.path.join(save_directory, "intrinsics.txt")
    if not os.path.exists(rgb_path):
        os.makedirs(rgb_path)
    if not os.path.exists(pose_path):
        os.makedirs(pose_path)
    
    cam_scale , cam_fx, cam_fy, cam_cx, cam_cy = 250, 333, 333, 120, 120
    K = torch.zeros((4,4), dtype= torch.float)
    K[0,0], K[1,1], K[2,2], K[0,2],K[1,2], cam_fx, cam_fy, 1, cam_cx, cam_cy
    K[0,0] = cam_fx
    K[1,1] = cam_fy
    K[2,2] = 1
    K[0,2] = cam_cx
    K[1,2] = cam_cy
    K[3,3] = 1
    np.savetxt(intrinsics_path, K)
    
    for folder in sorted(os.listdir(save_directory)):
        if not folder.startswith('.') and not folder.endswith('.txt'):
            for object_dir in (os.listdir(os.path.join(save_directory, folder))):
                if not object_dir.startswith('.'):
                    path = os.path.join(save_directory,folder, object_dir)
                    prefix = '0_'
                    colors = sorted(glob.glob(os.path.join(path, '*color.png')))
                    for color in colors:
                        name = os.path.basename(color)
                        if int(name.split("_")[0]) > 9:
                            file = rgb_path +'/' + prefix + '00' + name.split("_")[0] + ".png"
                        else:
                           file = rgb_path +'/' + prefix + '000' + name.split("_")[0] + ".png"
                        shutil.copyfile(color, file)
                    if not os.path.dirname(path).endswith('rgb') and not os.path.dirname(path).endswith('pose') and not os.path.basename(path).endswith('intrinsics.txt'):
                        with open(os.path.join(path,'poses.txt')) as f:
                            data = f.read()
                            data = data.split(';')
                            data.pop(-1) # remove empty entry
                            poses = []
                            for i in range(len(data)):
                                pose = np.eye(4)
                                js = json.loads(data[i])
                                number = js['pose']
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
                                if int(name.split("_")[0]) > 9:
                                    file = pose_path +'/' + prefix + '00' + str(number) + ".txt"
                                else:
                                    file = pose_path +'/' + prefix + '000' + str(number) + ".txt"
                                np.savetxt(file,pose)
                                
                            

if __name__ == "__main__":
    #dataset = BlenderDataset(save_directory  = '/home/maturk/data/test2')
    #dataset.get_object_paths()
    #object = dataset.__getitem__(0)
    #pc = torch.Tensor.numpy(object['depths'][0])
    #pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(pc)  
    #pc_gt = torch.Tensor.numpy(object['gt_pc'].detach().cpu()).transpose() 
    #pcdd = o3d.geometry.PointCloud()
    #pcdd.points = o3d.utility.Vector3dVector(pc_gt)  
    #o3d.visualization.draw_geometries([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame()]) 
    #shapenet_pc_sample(shapenet_directory = '/Users/maturk/data/Shapenet_small', save_directory = '/Users/maturk/data/test2',)
    blender_dataset_to_ngp_pl()