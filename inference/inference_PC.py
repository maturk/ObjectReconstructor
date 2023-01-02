'''
Run inference with a pretrained model for point cloud reconstruction
'''

import torch
import argparse
import os
from ObjectReconstructor.datasets import BlenderDataset
from ObjectReconstructor.models.point_ae import PointCloudAE, PointFusionAE
import numpy as np
from pytorch3d.loss import chamfer_distance
from torch.utils.data import random_split
import open3d as o3d
import csv
import pandas

#views_15_points_1024_emb_dim_256_epoch_1800, 
parser = argparse.ArgumentParser()
parser.add_argument('--visualize', type=bool, default=False, help='Set to true if you want to visualize registration for each test object')
parser.add_argument('--tsne', type=bool, default=True, help='Set to true if you want to visualize T-SNE embeddings')
parser.add_argument('--fusion', type=bool, default=False, help='Set to true if fusion model is to be used')
parser.add_argument('--num_points', type=int, default=1024, help='number of points in point cloud')
parser.add_argument('--emb_dim', type=int, default=256, help='dimension of latent embedding')
parser.add_argument('--device', type=str, default='cuda:0', help='GPU to use')
parser.add_argument('--load_model', type=str, default=f'{os.getcwd()}/results/point_clouds/models/views_15_points_1024_emb_dim_256_epoch_800.pth', help='load model path')
parser.add_argument('--result_dir', type=str, default=f'{os.getcwd()}/inference/results/point_clouds', help='directory to save inference results')
parser.add_argument('--save_dir', type=str, default=f'{os.getcwd()}/data/train_test_split/train', help='save directory of images to be used for inference (preprocessed shapenet images)')
parser.add_argument('--num_views', type=int, default=1, help = 'Number of input views per object instance, default set to 1 for single shot inference')


def inference(model, test_object):
    with torch.no_grad():
        test = test_object
        depths = test['depths'].clone().detach().to(device=opt.device, dtype= torch.float)
        
        with torch.cuda.amp.autocast(enabled=True):
            embedding, pc_out = model(depths)
            chamfer_loss = chamfer_distance(pc_out, test['gt_pc'].permute(0,2,1).to(device=opt.device, dtype= torch.float))[0]
            with open(f'{opt.result_dir}/results.csv',"a") as f:
                writer = csv.writer(f)
                writer.writerow([f'{int(test["class_dir"])}', f'{(test["object_dir"][0])}', f'{float(chamfer_loss)}'])
                
        #print("Predicted Reconstruction Chamfer Loss: ")
        #print(chamfer_loss)
       
        if opt.visualize: 
            # Visualize result    
            pc_out = pc_out.squeeze(0).cpu().detach().numpy()
            pc_gt = torch.Tensor.numpy(test['gt_pc']).transpose()
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
        model = PointFusionAE(emb_dim = opt.emb_dim, num_points= opt.num_points)
    else:
        model = PointCloudAE(emb_dim = opt.emb_dim, num_points= opt.num_points)
        
    try:
        model= torch.nn.DataParallel(model)
        model.to(opt.device)
        model.load_state_dict(torch.load(opt.load_model))
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
    
    test_dataset = BlenderDataset(mode = 'train', save_directory  = opt.save_dir, num_points=opt.num_points, num_views=opt.num_views)
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
                f.write('Class,Object,Chamfer Loss\n')
                
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
    print('Results: ')
    data = pandas.read_csv(f'{opt.result_dir}/results.csv')
    print('Chamfer Loss mean: ', data['Chamfer Loss'].mean())
                    
    
    
    
    