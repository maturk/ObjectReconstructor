import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ObjectReconstructor.datasets import BlenderDataset
from ObjectReconstructor.models.models import VoxelAE
import argparse
import numpy as np
from pytorch3d.loss import chamfer_distance
from torch.utils.data import random_split
import open3d as o3d
from pytorch_metric_learning import losses
import matplotlib.pyplot as plt

from ObjectReconstructor.configs import VOXEL_OCCUPANCY_TRESHOLD, VOXEL_RESOLUTION, VOXEL_SIZE
from ObjectReconstructor.utils import pytorch3d_vis


parser = argparse.ArgumentParser()
parser.add_argument('--voxel_size', type=int, default=128, help='voxel grid resolution')
parser.add_argument('--num_points', type=int, default=1024, help='voxel grid resolution')
parser.add_argument('--emb_dim', type=int, default=256, help='dimension of latent embedding')
parser.add_argument('--batch_size', type=int, default = 10, help='batch size')
parser.add_argument('--device', type=str, default='cuda:0', help='GPU to use')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=30, help='max number of epochs to train')
parser.add_argument('--load_model', type=str, default='', help='resume from saved model')
parser.add_argument('--result_dir', type=str, default='/home/maturk/git/ObjectReconstructor/results/voxels', help='directory to save train results')
parser.add_argument('--save_dir', type=str, default='/home/maturk/data/test2', help='save directory of preprocessed shapenet images')
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir')
parser.add_argument('--workers', '-w', type=int, default=1)
parser.add_argument('--num_views', type=int, default=15, help = 'Number of input views per object instance')


class Trainer():
    def __init__(self,
                 epochs,
                 device,
                 lr,
                 load_model,
                 results_dir,
                 batch_size,
                 num_views,
                 num_points,
                 voxel_size
                ):
        self.epochs = epochs
        self.device = device
        self.lr = lr
        self.load_model = load_model
        self.results_dir = results_dir
        self.batch_size = batch_size
        self.device = device
        self.num_views = num_views
        self.num_points = num_points
        self.voxel_size = voxel_size

    def train(self, model, train_dataloader, eval_dataloader):
        # TODO: implement full object trainer
        pass
    
    def train_one_object(self, model, object):
            model.train()
            loss = torch.nn.functional.binary_cross_entropy_with_logits
            gt_grid = object['gt_grid'].unsqueeze(0)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.99)
            depths = object['depths']
            xyzs = depths.clone().detach().to(device=self.device, dtype= torch.float32)
            xyzs = xyzs.unsqueeze(0)
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                embedding, grid_out = model(xyzs)
                out_loss = loss(grid_out, gt_grid.to(self.device))
                out_loss.backward()
                optimizer.step() 
                scheduler.step()
                
                if epoch % 50 == 0 and epoch!=0:
                    print('\n')
                    print('Epoch: ', epoch)
                    print('Loss: ', out_loss)
                    print('Learning rate: ', scheduler.get_last_lr())
                    if epoch % 500 == 0 and epoch!=0:
                        torch.save(model.state_dict(), f"{self.results_dir}/voxels_single_{self.num_views}_{self.voxel_size}_{epoch}.pth")
   
    def train_two_objects(self, model, objects):
        model.train()
        CE_loss = torch.nn.functional.binary_cross_entropy_with_logits
        contrastive_loss = losses.contrastive_loss.ContrastiveLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 500, gamma = 0.95)
        
        object1 = objects[0]
        object2 = objects[1]
        
        batch_depths_1 = object1['depths'].clone().detach().to(device=self.device, dtype= torch.float32)
        gt_pc_1 = object1['gt_grid'].clone().detach().to(device=self.device, dtype= torch.float32)
        batch_depths_2 = object2['depths'].clone().detach().to(device=self.device, dtype= torch.float32)
        gt_pc_2 = object2['gt_grid'].clone().detach().to(device=self.device, dtype= torch.float32)
        
        batch_nums = int(2)
        
        batch_depths = torch.cat((batch_depths_1, batch_depths_2), dim = 0)
        gt_grids = torch.cat((gt_pc_1, gt_pc_2), dim = 0)
        batch_depths = batch_depths.view(batch_nums, self.num_views, self.num_points, 3)
        gt_grids = gt_grids.view(batch_nums, VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION)
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            embedding, grid_out = model(batch_depths)
            out_CE_loss = CE_loss(grid_out, gt_grids.to(self.device))
            out_cont_loss = contrastive_loss(embedding, torch.Tensor([object1['class_dir'], object2['class_dir']]))
            
            out_loss = out_CE_loss + out_cont_loss
            out_loss.backward()
            optimizer.step() 
            scheduler.step()
            
            if epoch % 50 == 0 and epoch!=0:
                print('\n')
                print('Epoch: ', epoch)
                print('Loss: ', out_loss)
                print('CE loss: ', out_CE_loss, 'contrastive loss: ', out_cont_loss)
                print('norm 1 : ', embedding[0].norm(), 'norm 2 : ', embedding[1].norm())
                print('mean 1: ', embedding[0].mean(), 'mean 2: ', embedding[1].mean() )
                print('Embedding norm : ', embedding.norm(), 'Embedding mean: ', embedding.mean() )
                print('Learning rate: ', scheduler.get_last_lr())
                if epoch % 500 == 0 and epoch!=0:
                    torch.save(model.state_dict(), f"{self.results_dir}/voxels_double_{self.num_views}_{self.voxel_size}_{epoch}.pth")
        
        
def main():
    # TODO: Implement full object trainer
    pass

def train_one_object():
    opt = parser.parse_args() 
    opt.epochs = 1001
    dataset = BlenderDataset(mode = 'train', save_directory  = '/home/maturk/data/small_set', voxel_size=opt.voxel_size)
    object = dataset.__getitem__(0)
    model = VoxelAE(emb_dim = opt.emb_dim, voxel_size= opt.voxel_size).to(opt.device)
    trainer = Trainer(opt.epochs, opt.device, opt.lr, opt.load_model, opt.result_dir, opt.batch_size, opt.num_views, opt.num_points, opt.voxel_size)
    trainer.train_one_object(model, object)
    
def train_two_objects():
    opt = parser.parse_args() 
    opt.epochs = 2501
    dataset = BlenderDataset(mode = 'train', save_directory  = '/home/maturk/data/small_set', voxel_size=opt.voxel_size)
    object_0 = dataset.__getitem__(0)
    object_1 = dataset.__getitem__(1)
    
    model = VoxelAE(emb_dim = opt.emb_dim, voxel_size= opt.voxel_size).to(opt.device)
    trainer = Trainer(opt.epochs, opt.device, opt.lr, opt.load_model, opt.result_dir, opt.batch_size, opt.num_views, opt.num_points, opt.voxel_size)
    trainer.train_two_objects(model, [object_0, object_1])
    

def test_one_object():
    import os
    opt = parser.parse_args()
    dataset = BlenderDataset(mode = 'train', save_directory  = '/home/maturk/data/small_set')
    model = VoxelAE(emb_dim = opt.emb_dim, voxel_size= opt.voxel_size).to(opt.device)
    model.load_state_dict(torch.load(os.path.join(opt.result_dir,f'voxels_single_10_{opt.voxel_size}_1000.pth')))
    
    # Choose item to test (same as trained object)
    object = dataset.__getitem__(0)
    gt_grid = object['gt_grid']
    depths = object['depths']
    xyzs = depths.clone().detach().to(device=opt.device, dtype= torch.float32).unsqueeze(0)
    
    embedding, out_grid = model(xyzs)
    out_grid = out_grid.squeeze(0)
    voxel_probs = out_grid.sigmoid() 
    
    # Loss wrt. MSELoss or CELoss
    mse_loss = torch.nn.MSELoss()
    ce_loss = torch.nn.functional.binary_cross_entropy_with_logits
    out_mse_loss = mse_loss(voxel_probs, object['gt_grid'].to(opt.device))
    out_ce_loss = ce_loss(out_grid, object['gt_grid'].to(opt.device))
    print('MSE Loss: ', out_mse_loss, 'CE Loss: ', out_ce_loss)
    
    PYTORCH3D_VIS = True
    OBJECT_NAME = 'bottle'
    if PYTORCH3D_VIS == True:
        from PIL import Image
        images = pytorch3d_vis(gt_grid.unsqueeze(0), view_elev=0)
        images = images[0, ..., :3].cpu().numpy() *255
        images = images.astype(np.uint8)
        im = Image.fromarray(images)
        im.save(os.path.join(opt.result_dir, 'images', f'gt_{OBJECT_NAME}.png' ))
        
        # render a few different predicted images based on elevation
        for elev in range(0,180,40):
            images = pytorch3d_vis(voxel_probs.unsqueeze(0), view_elev=elev)
            images = images[0, ..., :3].cpu().numpy() *255
            images = images.astype(np.uint8)
            im = Image.fromarray(images)
            im.save(os.path.join(opt.result_dir, 'images', f'pred_{OBJECT_NAME}_{elev}.png' )) 
            
def test_two_objects():
    import os
    opt = parser.parse_args()
    dataset = BlenderDataset(mode = 'train', save_directory  = '/home/maturk/data/small_set')
    model = VoxelAE(emb_dim = opt.emb_dim, voxel_size= opt.voxel_size).to(opt.device)
    model.load_state_dict(torch.load(os.path.join(opt.result_dir,f'voxels_double_{opt.num_views}_{opt.voxel_size}_2500.pth')))
    
    # Choose item to test (same as trained object)
    object_0 = dataset.__getitem__(0)
    object_1 = dataset.__getitem__(1)
    
    gt_grid_0 = object_0['gt_grid']
    depths_0 = object_0['depths']
    xyzs_0 = depths_0.clone().detach().to(device=opt.device, dtype= torch.float32).unsqueeze(0)
    
    gt_grid_1 = object_1['gt_grid']
    depths_1 = object_1['depths']
    xyzs_1 = depths_1.clone().detach().to(device=opt.device, dtype= torch.float32).unsqueeze(0)
    
    embedding, out_grid_0 = model(xyzs_0)
    embedding, out_grid_1 = model(xyzs_1)
    
    out_grid_0 = out_grid_0.squeeze(0)
    voxel_probs_0 = out_grid_0.sigmoid() 
    out_grid_1 = out_grid_1.squeeze(0)
    voxel_probs_1 = out_grid_1.sigmoid() 
    
    # Loss wrt. MSELoss or CELoss
    mse_loss = torch.nn.MSELoss()
    ce_loss = torch.nn.functional.binary_cross_entropy_with_logits
    out_mse_loss = mse_loss(voxel_probs_0, gt_grid_0.to(opt.device))
    out_ce_loss = ce_loss(out_grid_0, gt_grid_0.to(opt.device))
    print('Losses for object 0: ')
    print('MSE Loss: ', out_mse_loss, 'CE Loss: ', out_ce_loss)
    out_mse_loss = mse_loss(voxel_probs_1, gt_grid_1.to(opt.device))
    out_ce_loss = ce_loss(out_grid_1, gt_grid_1.to(opt.device))
    print('Losses for object 1: ')
    print('MSE Loss: ', out_mse_loss, 'CE Loss: ', out_ce_loss)
    
    
    PYTORCH3D_VIS = True
    OBJECT_NAME_0 = 'bottle'
    OBJECT_NAME_1 = 'cup'
    if PYTORCH3D_VIS == True:
        from PIL import Image
        images = pytorch3d_vis(gt_grid_0.unsqueeze(0), view_elev=0)
        images = images[0, ..., :3].cpu().numpy() *255
        images = images.astype(np.uint8)
        im = Image.fromarray(images)
        im.save(os.path.join(opt.result_dir, 'images', f'gt_{OBJECT_NAME_0}.png' ))
        # render a few different predicted images based on elevation
        for elev in range(0,180,40):
            images = pytorch3d_vis(voxel_probs_0.unsqueeze(0), view_elev=elev)
            images = images[0, ..., :3].cpu().numpy() *255
            images = images.astype(np.uint8)
            im = Image.fromarray(images)
            im.save(os.path.join(opt.result_dir, 'images', f'pred_double_{OBJECT_NAME_0}_{elev}.png' )) 
    if PYTORCH3D_VIS == True:
        from PIL import Image
        images = pytorch3d_vis(gt_grid_1.unsqueeze(0), view_elev=0)
        images = images[0, ..., :3].cpu().numpy() *255
        images = images.astype(np.uint8)
        im = Image.fromarray(images)
        im.save(os.path.join(opt.result_dir, 'images', f'gt_{OBJECT_NAME_1}.png' ))
        # render a few different predicted images based on elevation
        for elev in range(0,180,40):
            images = pytorch3d_vis(voxel_probs_1.unsqueeze(0), view_elev=elev)
            images = images[0, ..., :3].cpu().numpy() *255
            images = images.astype(np.uint8)
            im = Image.fromarray(images)
            im.save(os.path.join(opt.result_dir, 'images', f'pred_double_{OBJECT_NAME_1}_{elev}.png' )) 
            

    
if __name__ == "__main__":
    #train_one_object()
    #test_one_object()
    #train_two_objects()
    test_two_objects()
    