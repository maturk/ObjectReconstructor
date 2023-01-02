'''
Run inference with a pretrained model for voxel reconstruction
'''


import torch
import argparse
import os
import numpy as np
from pytorch3d.loss import chamfer_distance
import open3d as o3d
import csv
from pytorch_metric_learning import losses
from ObjectReconstructor.datasets import BlenderDataset
from ObjectReconstructor.models.voxel_ae import VoxelAE, VoxelFusionAE
from ObjectReconstructor.configs import VOXEL_OCCUPANCY_TRESHOLD, VOXEL_RESOLUTION, VOXEL_SIZE
from ObjectReconstructor.utils import pytorch3d_vis, voxel_to_pc

#cluster_views_15_emb_dim_1024_voxel_size_128_lr_0.001_epoch_1800.pth latest large train. 2300 also okay
parser = argparse.ArgumentParser()
parser.add_argument('--visualize', type=bool, default=False, help='Set to true if you want to visualize registration for each test object')
parser.add_argument('--tsne', type=bool, default=False, help='Set to true if you want to visualize T-SNE embeddings')
parser.add_argument('--fusion', type=bool, default=False, help='Set to true if fusion model is to be used')
parser.add_argument('--voxel_size', type=int, default=128, help='voxel grid resolution')
parser.add_argument('--emb_dim', type=int, default=1024, help='dimension of latent embedding')
parser.add_argument('--device', type=str, default='cuda', help='GPU to use')
parser.add_argument('--load_model', type=str, default=f'{os.getcwd()}/results/voxels/models/cluster_views_15_emb_dim_1024_voxel_size_128_lr_0.001_epoch_2300.pth', help='load model path')
parser.add_argument('--result_dir', type=str, default=f'{os.getcwd()}/inference/results/voxels', help='directory to save inference results')
parser.add_argument('--save_dir', type=str, default=f'{os.getcwd()}/data/train_test_split/train', help='save directory of images to be used for inference (preprocessed shapenet images)')
parser.add_argument('--num_views', type=int, default=1, help = 'Number of input views per object instance, default set to 1 for single shot inference')


def inference(model, test_object):
    CE_loss = torch.nn.functional.binary_cross_entropy_with_logits
    PC_loss = voxel_to_pc
    data = test_object
    with torch.no_grad():
        batch_depths = data['depths'].clone().detach().to(device=opt.device, dtype= torch.float32)
        gt_grids = data['gt_grid'].to(device = opt.device, dtype = torch.float32)
        
        if opt.fusion:
            batch_colors = data['colors'].clone().detach().to(device=opt.device, dtype= torch.float32)
            batch_masks = data['masks'].clone().detach().to(device=opt.device, dtype= torch.int64)
            embedding, grid_out = model(colors = batch_colors, xyzs = batch_depths, masks = batch_masks)
        else: embedding, grid_out = model(batch_depths)
        
        out_CE_loss = CE_loss(grid_out, gt_grids)
        with open(f'{opt.result_dir}/results.csv',"a") as f:
            writer = csv.writer(f)
            writer.writerow([f'{int(data["class_dir"])}', f'{(data["object_dir"][0])}', f'{float(out_CE_loss)}'])
                
        #print("Predicted Reconstruction Chamfer Loss: ")
        #print(chamfer_loss)
       
        if opt.visualize: 
            # Visualize result    
            pc_out = pc_out.squeeze(0).cpu().detach().numpy()
            pc_gt = torch.Tensor.numpy(data['gt_pc']).transpose()
            pcd_gt = o3d.geometry.PointCloud()
            pcd_out = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(pc_gt)
            pcd_out.points = o3d.utility.Vector3dVector(pc_out)
            pcd_gt.colors = o3d.utility.Vector3dVector()
            o3d.visualization.draw_geometries([pcd_out])
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
    
    test_dataset = BlenderDataset(mode = 'train', save_directory  = opt.save_dir, num_views=opt.num_views, voxel_size=opt.voxel_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=1,
                                                   num_workers=1,
                                                   shuffle = False)
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
                f.write('Class,Object,Cross Entropy Loss,Chamfer Loss \n')
                
        _ = inference(model,test_object)
        if opt.tsne:
            embeddings = torch.cat((embeddings, _.detach().cpu()), 0)
            labels.append(int(np.array(test_object['class_dir'])))
            if test_object['class_dir'] not in categories:
                categories.append(test_object['class_dir'])
    
    # Plot T-SNE embedding space for dataset
    if opt.tsne:
        embeddings = np.array(embeddings)
        if embeddings.shape[0]<=20:
            tsne = TSNE(2, perplexity=embeddings.shape[0]-1, verbose=0)
        else: tsne = TSNE(2, perplexity=20, verbose=0)
        tsne_proj = tsne.fit_transform(embeddings)
        cmap = cm.get_cmap('tab20')
        fig, ax = plt.subplots(figsize=(8,8))
        num_categories = len(categories)
        labels = np.array(labels)
        for g in np.unique(labels):
            ix = np.where(labels == g)
            ax.scatter(tsne_proj[ix,0], tsne_proj[ix,1], label = g, s = 100)
        ax.legend(fontsize='large', markerscale=2)
        plt.show()
    print(f'Inference finished: results saved to {opt.result_dir}/results.csv')
    print('------------------------------------------------------------------------------------------------------------------------------------')