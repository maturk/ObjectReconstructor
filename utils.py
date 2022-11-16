import torch
import os
import glob
import numpy as np
import open3d as o3d
import json
from scipy.spatial.transform import Rotation 
from ObjectReconstructor.configs import VOXEL_OCCUPANCY_TRESHOLD, VOXEL_RESOLUTION, VOXEL_SIZE, CUBIC_SIZE, GRID_MAX, GRID_MIN


def voxel_to_pc(voxel):
    voxel = voxel.sigmoid()
    out = []
    for i in range(voxel.shape[0]):
        indices = (voxel[i,:] >= 0.5).nonzero(as_tuple=False)
        if indices.nelement() == 0:
            points = torch.zeros((1,3))
        else:
            points = VOXEL_SIZE * indices[:,:] - CUBIC_SIZE / 2
        out.append(points)
    return out

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
                        # store ground truth PC
                        model = models[0]
                        mesh = o3d.io.read_triangle_mesh(model)
                        points = mesh.sample_points_uniformly(sample_points)
                        o3d.io.write_point_cloud(os.path.join(save_directory, folder, object_dir, 'pc.pcd'), points, compressed = False)
                        # store ground truth voxel grid
                        print(os.path.join(save_directory, folder, object_dir))
                        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size = VOXEL_SIZE, min_bound = GRID_MIN, max_bound = GRID_MAX)
                        voxels = voxel_grid.get_voxels() 
                        indicies = np.stack(list(vx.grid_index for vx in voxels))
                        grid = np.zeros((VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION), dtype= np.int32)
                        grid[tuple(indicies.T)]=1
                        np.save(os.path.join(save_directory, folder, object_dir, 'grid.npy'), grid)
                        print('saved')
                        counter += 1
                    else:
                        continue


def blender_dataset_to_ngp_pl(save_directory):
    ''' Convert blender dataset file format to ngp_pl file format
    
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

                               
def pytorch3d_vis(grid, view_elev = 0.0, thresh = VOXEL_OCCUPANCY_TRESHOLD):
    ''' Pytorch3D visualizer for untextured voxel grids
    
    Args:
        grid (tensor): voxel grid tensor with normalized (sigmoid act) occupancy probabilities.
        view_elev (float): elevation angle of rendered image
        tresh (float): occupancy threshold value
    '''
    
    import matplotlib.pyplot as plt
    from pytorch3d.ops import cubify
    # Data structures and functions for rendering
    from pytorch3d.structures import Meshes
    from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
    from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
    from pytorch3d.renderer import (
        look_at_view_transform,
        FoVPerspectiveCameras, 
        PointLights, 
        RasterizationSettings, 
        MeshRenderer, 
        MeshRasterizer,  
        SoftPhongShader,
        TexturesVertex )
    
    mesh = cubify(grid, thresh=thresh) # voxel to mesh
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

    R, T = look_at_view_transform(1.5, view_elev, 180) 
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = RasterizationSettings(
        image_size=512, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        ))

    verts = mesh.verts_list()[0]
    faces = mesh.faces_list()[0]
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    render_mesh =  Meshes(
    verts=[verts.to(device)],
    faces=[faces.to(device)],
    textures=textures)

    images = renderer(render_mesh)
    
    return images