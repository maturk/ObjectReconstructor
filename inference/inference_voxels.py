'''
Run inference with a pretrained model for voxel reconstruction
'''


import torch
import argparse
import os
import numpy as np
import open3d as o3d
import csv
from ObjectReconstructor.datasets import BlenderDataset
from ObjectReconstructor.models.voxel_ae import VoxelAE, VoxelFusionAE
from ObjectReconstructor.configs import VOXEL_OCCUPANCY_TRESHOLD, VOXEL_RESOLUTION, VOXEL_SIZE
from ObjectReconstructor.utils import pytorch3d_vis, voxel_to_pc, voxel_IoU, pc_down_sample
from ObjectReconstructor.scratch import vis
from pytorch3d.loss import chamfer_distance
import pandas as pd
from PIL import Image
from torch.utils.data import random_split


parser = argparse.ArgumentParser()
parser.add_argument('--visualize', type=bool, default=False, help='Set to true if you want to visualize registration for each test object')
parser.add_argument('--save_renders', type=bool, default=False, help='Render voxel grid reconstructions and save them to results folder')
parser.add_argument('--save_pc', type=bool, default=False, help='Render voxel grid reconstructions and save them to results folder')
parser.add_argument('--tsne', type=bool, default=False, help='Set to true if you want to visualize T-SNE embeddings')
parser.add_argument('--fusion', type=bool, default=False, help='Set to true if fusion model is to be used')
parser.add_argument('--voxel_size', type=int, default=128, help='voxel grid resolution')
parser.add_argument('--emb_dim', type=int, default=1024, help='dimension of latent embedding')
parser.add_argument('--device', type=str, default='cuda', help='GPU to use')
parser.add_argument('--load_model', type=str, default=f'{os.getcwd()}/results/voxels/models/50%_cluster_views_15_emb_dim_1024_voxel_size_128_lr_0.001_epoch_1440.pth', help='load model path')
parser.add_argument('--result_dir', type=str, default=f'{os.getcwd()}/inference/results/voxels', help='directory to save inference results')
parser.add_argument('--save_dir', type=str, default=f'{os.getcwd()}/data/large_train_test_split/test', help='save directory of images to be used for inference (preprocessed shapenet images)')
parser.add_argument('--num_views', type=int, default=5, help = 'Number of input views per object instance, default set to 1 for single shot inference')


names = {'03211117':'display', '02992529':'cellphone', '02954340':'cap', '02942699':'camera', '02946921':'can', '03797390':'mug', '02880940':'bowl',
         '3211117':'display', '2992529':'cellphone', '2954340':'cap', '2942699':'camera', '2946921':'can', '3797390':'mug', '2880940':'bowl', '2876657':'bottle', '3642806':'laptop'}


def inference(model, test_object):
    CE_loss = torch.nn.functional.binary_cross_entropy_with_logits
    data = test_object
    with torch.no_grad():
        batch_depths = data['depths'].clone().detach().to(device=opt.device, dtype= torch.float32)
        gt_grids = data['gt_grid'].to(device = opt.device, dtype = torch.float32)
        
        if opt.fusion:
            batch_colors = data['colors'].clone().detach().to(device=opt.device, dtype= torch.float32)
            batch_masks = data['masks'].clone().detach().to(device=opt.device, dtype= torch.int64)
            embedding, grid_out = model(colors = batch_colors, xyzs = batch_depths, masks = batch_masks) # grid_out = unnormalized logits
        else: embedding, grid_out = model(batch_depths)
        
        # Losses: Cross Entropy, Chamfer Loss, and 3D Voxel IoU
        out_CE_loss = CE_loss(grid_out, gt_grids)
        pc_out = voxel_to_pc(grid_out.sigmoid())[0].detach()
        chamfer_loss = chamfer_distance(pc_out.unsqueeze(0), data['gt_pc'].permute(0,2,1).to(device=opt.device, dtype= torch.float))[0]
        IoU = voxel_IoU(grid_out.sigmoid().detach().cpu().numpy(), gt_grids.cpu().numpy())
        
        with open(f'{opt.result_dir}/results.csv',"a") as f:
            writer = csv.writer(f)
            writer.writerow([f'{names[str(int(data["class_dir"]))]}', f'{(data["object_dir"][0])}', f'{float(out_CE_loss)}', f'{float(chamfer_loss)}', f'{IoU}'])
                
        if opt.visualize: 
            # Visualize result    
            pc_out = pc_out.squeeze(0).cpu().detach().numpy()
            pc_gt = torch.Tensor.numpy(data['gt_pc']).transpose()
            pcd_gt = o3d.geometry.PointCloud()
            pcd_out = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(pc_gt)
            pcd_out.points = o3d.utility.Vector3dVector(pc_out)
            pcd_gt.colors = o3d.utility.Vector3dVector()
            pcd_gt.paint_uniform_color([0, 0, 0])
            pcd_out.paint_uniform_color([0, 1, 0])
            o3d.visualization.draw_geometries([pcd_gt,pcd_out])
            
        if opt.save_renders:
            images = pytorch3d_vis(grid_out.sigmoid(), view_elev=20, azimuth=180)
            images = (images[0, ..., :3].cpu().numpy() *255).astype(np.uint8)
            im = Image.fromarray(images)
            if not os.path.exists(os.path.join(opt.result_dir, 'images', 'out_voxels')):
                os.makedirs(os.path.join(opt.result_dir, 'images', 'out_voxels'))
            im.save(os.path.join(opt.result_dir, 'images', 'out_voxels', f'pred_angle_{i}_{opt.num_views}_{names[str(int(data["class_dir"]))]}_{str(data["object_dir"][0])}.png' ))
            print('saved prediction: ', os.path.join(opt.result_dir, 'images','out_voxels', f'pred_angle_{i}_{opt.num_views}_{names[str(int(data["class_dir"]))]}_{data["object_dir"][0]}.png'))
            
            gt = pytorch3d_vis(gt_grids, view_elev=20, azimuth=180)
            gt = (gt[0, ..., :3].cpu().numpy() *255).astype(np.uint8)
            im = Image.fromarray(gt)
            if not os.path.exists(os.path.join(opt.result_dir, 'images')):
                os.makedirs(os.path.join(opt.result_dir, 'images'))
            im.save(os.path.join(opt.result_dir, 'images', 'out_voxels',f'gt_angle_{i}_{names[str(int(data["class_dir"]))]}_{str(data["object_dir"][0])}.png' ))
            print('saved gt: ', os.path.join(opt.result_dir, 'images','out_voxels', f'gt_angle_{i}_{names[str(int(data["class_dir"]))]}_{data["object_dir"][0]}.png'))
            
        if opt.save_pc:
            pc_out = pc_out.squeeze(0).cpu().detach().numpy()
            pc_out ,_ = pc_down_sample(pc_out,2048)
            pc_gt = data['gt_pc'].squeeze(0).numpy().transpose()
            if not os.path.exists(os.path.join(opt.result_dir, 'images', 'point_clouds')):
                os.makedirs(os.path.join(opt.result_dir, 'images', 'point_clouds'))
            vis.showpoints(pc_out)
            for i in range(0,360,20):
                vis.renderpoints(pc_out, zangle = i, path = os.path.join(opt.result_dir, 'images', 'point_clouds', f'pred_angle_{i}_{names[str(int(data["class_dir"]))]}_{data["object_dir"][0]}.png'))
            print('saved prediction: ', os.path.join(opt.result_dir, 'images', 'point_clouds', f'pred_{names[str(int(data["class_dir"]))]}_{data["object_dir"][0]}.png'))
            vis.renderpoints(pc_gt, path = os.path.join(opt.result_dir, 'images', 'point_clouds', f'gt_{names[str(int(data["class_dir"]))]}_{data["object_dir"][0]}.png'))
            print('saved gt: ', os.path.join(opt.result_dir, 'images', 'point_clouds' ,f'gt_{names[str(int(data["class_dir"]))]}_{data["object_dir"][0]}.png'))
            
    return embedding
    
    
if __name__ == "__main__":
    opt = parser.parse_args() 
    
    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)
        
    if opt.fusion == True:
        model = VoxelFusionAE(emb_dim = opt.emb_dim, voxel_size= opt.voxel_size)
    else:
        model = VoxelAE(emb_dim = opt.emb_dim, voxel_size= opt.voxel_size)
    
    try:
        model.load_state_dict(torch.load(opt.load_model))
        model.to(opt.device)
        model.eval()
        print('------------------------------------------------------------------------------------------------------------------------------------')
        print('Loading model worked')
        print('------------------------------------------------------------------------------------------------------------------------------------')
    except:
        print("ERROR: Invalid model path OR Invalid state_dict... trying without DataParallel")
        model= torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(opt.load_model))
        model.to(opt.device)
        model.eval()
        print('------------------------------------------------------------------------------------------------------------------------------------')
        print('Loading model worked')
        print('------------------------------------------------------------------------------------------------------------------------------------')
    
    test_dataset = BlenderDataset(mode = 'train', save_directory  = opt.save_dir, voxel_size=opt.voxel_size, num_views= opt.num_views)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=1,
                                                   num_workers=1,
                                                   shuffle = False)
    print('Current dataset len: ', test_dataset.__len__())
    
    if opt.tsne:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from matplotlib import cm
        embeddings = torch.zeros((0, opt.emb_dim), dtype=torch.float32)
        labels = []
        categories = []
    
    for i, test_object in enumerate(test_dataloader):
        if i == 0: 
            if os.path.isfile(f'{opt.result_dir}/results.csv'):
                os.remove(f'{opt.result_dir}/results.csv')
            with open(f'{opt.result_dir}/results.csv',"w+") as f:
                f.write('Class,Object,Cross Entropy Loss,Chamfer Loss,IoU\n')
                
        _ = inference(model,test_object)
        if opt.tsne:
            embeddings = torch.cat((embeddings, _.detach().cpu()), 0)
            labels.append(int(np.array(test_object['class_dir'])))
            if test_object['class_dir'] not in categories:
                categories.append(test_object['class_dir'])
    
    np.save(f'{opt.result_dir}/tsne', np.array(embeddings), allow_pickle=True)
    np.save(f'{opt.result_dir}/categories', np.array(labels), allow_pickle=True)
    # Plot T-SNE embedding space for dataset
    if opt.tsne:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from matplotlib import cm
        embeddings = np.load(f'{opt.result_dir}/tsne.npy')
        labels = np.load(f'{opt.result_dir}/categories.npy')
        embeddings = np.array(embeddings)
        print('Num of embeddings: ', embeddings.shape[0])
        if embeddings.shape[0]<=20:
            tsne = TSNE(2, perplexity=embeddings.shape[0]-1, verbose=0)
        else: tsne = TSNE(2, perplexity=25, verbose=1) #75
        tsne_proj = tsne.fit_transform(embeddings)
        cmap = cm.get_cmap('tab20')
        fig, ax = plt.subplots(figsize=(8,8))
        labels = np.array(labels)
        for g in np.unique(labels):
            ix = np.where(labels == g)
            ax.scatter(tsne_proj[ix,0], tsne_proj[ix,1], label = names[str(g)], s = 100)
        ax.legend(fontsize='large', markerscale=2)
        plt.savefig(f'{opt.result_dir}/tsne.png')
    print(f'Inference finished: results saved to {opt.result_dir}/results.csv')
    print('------------------------------------------------------------------------------------------------------------------------------------')
    data = pd.read_csv(f'{opt.result_dir}/results.csv')
    print('Mean Cross-Entropy Loss: ', data['Cross Entropy Loss'].mean())
    print('Mean Chamfer Loss: ', data['Chamfer Loss'].mean())
    print('Mean IoU: ', data['IoU'].mean()) 
    print(data.groupby(by='Class').mean())