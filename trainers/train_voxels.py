import torch
from torch.utils.data import random_split
from ObjectReconstructor.datasets import BlenderDataset
from ObjectReconstructor.models.voxel_ae import VoxelAE, VoxelFusionAE
import argparse
import os
import numpy as np
from pytorch3d.loss import chamfer_distance
from torch.utils.data import random_split
from pytorch_metric_learning import losses
from ObjectReconstructor.configs import VOXEL_OCCUPANCY_TRESHOLD, VOXEL_RESOLUTION, VOXEL_SIZE
from ObjectReconstructor.utils import pytorch3d_vis, voxel_to_pc
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--fusion', type=bool, default=True, help='Set to true if fusion model is to be used. Fusio combines both RGB and Depth channel for encoder. Default False, Depth only encoder.')
parser.add_argument('--voxel_size', type=int, default=128, help='voxel grid resolution')
parser.add_argument('--emb_dim', type=int, default=1024, help='dimension of latent embedding')
parser.add_argument('--batch_size', type=int, default =1, help='batch size')
parser.add_argument('--device', type=str, default='cuda', help='GPU to use')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=30, help='max number of epochs to train')
parser.add_argument('--load_model', type=str, default='', help='resume from saved model')
parser.add_argument('--result_dir', type=str, default='/home/maturk/git/ObjectReconstructor/results/voxels', help='directory to save train results')
parser.add_argument('--log_dir', type=str, default='/home/maturk/git/ObjectReconstructor/results/voxels/logs', help='directory to save train log results')
parser.add_argument('--save_dir', type=str, default='/home/maturk/data/test2', help='save directory of preprocessed shapenet images')
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
                 voxel_size,
                 log_dir,
                 emb_dim,
                 fusion
                ):
        self.epochs = epochs
        self.device = device
        self.lr = lr
        self.load_model = load_model
        self.results_dir = results_dir
        self.batch_size = batch_size
        self.device = device
        self.num_views = num_views
        self.voxel_size = voxel_size
        self.log_dir = log_dir
        self.emb_dim = emb_dim
        self.fusion = fusion

    def train(self, model, train_dataloader, eval_dataloader):
        writer = SummaryWriter(log_dir=self.log_dir)
        CE_loss = torch.nn.functional.binary_cross_entropy_with_logits
        contrastive_loss = losses.contrastive_loss.ContrastiveLoss()
        PC_loss = voxel_to_pc
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 500, gamma = 0.95)
        
        for epoch in tqdm(range(self.epochs)):
            model.train()
            batch_idx = 0
            for i, data in enumerate(train_dataloader):
                optimizer.zero_grad()
                batch_depths = data['depths'].clone().detach().to(device=self.device, dtype= torch.float32)
                gt_grids = data['gt_grid'].to(device = self.device, dtype = torch.float32)
                
                if self.fusion:
                    batch_colors = data['colors'].clone().detach().to(device=self.device, dtype= torch.float32)
                    batch_masks = data['masks'].clone().detach().to(device=self.device, dtype= torch.int64)
                    embedding, grid_out = model(colors = batch_colors, xyzs = batch_depths, masks = batch_masks)
                else: embedding, grid_out = model(batch_depths)
                
                out_CE_loss = CE_loss(grid_out, gt_grids)
                out_cont_loss = contrastive_loss(embedding, torch.Tensor(data['class_dir']))
                out_loss = out_CE_loss + out_cont_loss 
                out_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5) 

                optimizer.step() 
                scheduler.step()
                
                if epoch % 1 == 0 and batch_idx % 5 == 0:
                    print('\n')
                    print('Epoch: ', epoch, 'Batch idx: ', batch_idx)
                    print('Loss: ', out_loss)
                    print('CE loss: ', out_CE_loss, 'contrastive loss: ', out_cont_loss)
                    #print('Embedding norm : ', embedding.norm(), 'Embedding mean: ', embedding.mean() )
                    #print('Learning rate: ', scheduler.get_last_lr())
                    writer.add_scalar("Loss / train", out_loss, epoch)
                    writer.add_scalar("CE Loss / train", out_CE_loss, epoch)
                    writer.add_scalar("Contrastive Loss / train", out_cont_loss, epoch)
                    writer.add_scalar("Lr",scheduler.get_last_lr()[0], epoch)
                    
                    if epoch % 200 == 0 and epoch!=0:
                        torch.save(model.state_dict(), f"{self.results_dir}/models/voxels_multi_views_{self.num_views}_emb_dim_{self.emb_dim}_voxel_size_{self.voxel_size}_lr_{self.lr}_epoch_{epoch}.pth")
                batch_idx+=1
        writer.flush()
        
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
                        torch.save(model.state_dict(), f"{self.results_dir}/voxels_single_{self.num_views}_{self.voxel_size}_{self.lr}_{epoch}.pth")
   
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
            
            if epoch % 5 == 0 and epoch!=0:
                print('\n')
                print('Epoch: ', epoch)
                print('Loss: ', out_loss)
                print('CE loss: ', out_CE_loss, 'contrastive loss: ', out_cont_loss)
                print('Learning rate: ', scheduler.get_last_lr())
                if epoch % 1000 == 0 and epoch!=0:
                    torch.save(model.state_dict(), f"{self.results_dir}/models/two_voxels_double_views_{self.num_views}_emb_dim_{self.emb_dim}_voxel_size_{self.voxel_size}_lr_{self.lr}_epoch_{epoch}.pth")
                    
    def train_n_objects(self, model, objects):
        writer = SummaryWriter(log_dir=self.log_dir)
        model.train()
        CE_loss = torch.nn.functional.binary_cross_entropy_with_logits
        contrastive_loss = losses.contrastive_loss.ContrastiveLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 500, gamma = 1)
        
        batch_depths = []
        gt_pc = []
        classes = []
        for n in range(len(objects)):
            batch_depths.append(objects[n]['depths'].clone().detach().to(device=self.device, dtype= torch.float32))
            gt_pc.append(objects[n]['gt_grid'].clone().detach().to(device=self.device, dtype= torch.float32))
            classes.append(objects[n]['class_dir'])
        
        batch_depths = torch.stack([depth for depth in batch_depths], dim = 0)
        gt_grids = torch.stack([gt for gt in gt_pc], dim = 0)
        
        for epoch in tqdm(range(self.epochs)):
            optimizer.zero_grad()
            embedding, grid_out = model(batch_depths)
            out_CE_loss = CE_loss(grid_out, gt_grids.to(self.device))
            out_cont_loss = contrastive_loss(embedding, torch.Tensor(classes))
            
            out_loss = out_CE_loss + out_cont_loss
            out_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step() 
            scheduler.step()
            
            if epoch % 50 == 0 and epoch!=0:
                print('\n')
                print('Epoch: ', epoch)
                print('Loss: ', out_loss)
                print('CE loss: ', out_CE_loss, 'contrastive loss: ', out_cont_loss)
                print('Learning rate: ', scheduler.get_last_lr())
                writer.add_scalar("Loss / train", out_loss, epoch)
                writer.add_scalar("CE Loss / train", out_CE_loss, epoch)
                writer.add_scalar("Contrastive Loss / train", out_cont_loss, epoch)
                writer.add_scalar("Lr",scheduler.get_last_lr()[0], epoch)
                if epoch % 500 == 0 and epoch!=0:
                    torch.save(model.state_dict(), f"{self.results_dir}/models/{len(objects)}_models__voxels_views_{self.num_views}_emb_dim_{self.emb_dim}_voxel_size_{self.voxel_size}_lr_{self.lr}_epoch_{epoch}.pth")
        writer.flush()


def main():
    opt = parser.parse_args() 
    opt.epochs = 2501
    opt.lr = 0.001
    opt.load_model = '' #'/cluster/home/mturkulainen/ObjectReconstructor/results/models/cluster_views_15_emb_dim_1024_voxel_size_128_lr_0.001_epoch_460.pth'
    train_dataset = BlenderDataset(mode = 'train', save_directory  = '/home/maturk/data/train_test_split/train', voxel_size=opt.voxel_size)
    eval_dataset = BlenderDataset(mode = 'train', save_directory  = '/home/maturk/data/train_test_split/test', voxel_size=opt.voxel_size)
    print('Current train and eval split: ', train_dataset.__len__(), eval_dataset.__len__()) 
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=opt.batch_size,
                                                   num_workers=4,
                                                   shuffle = True,
                                                   pin_memory=True)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                   batch_size=opt.batch_size,
                                                   num_workers=opt.workers,
                                                   shuffle = False)
        
    if opt.fusion:
        print('Using Fusion Voxel Auto-Encoder network')
        model = VoxelFusionAE(emb_dim = opt.emb_dim, voxel_size= opt.voxel_size)
    else:
        print('Using PointNet (Depth only) Voxel Auto-Encoder network')
        model = VoxelAE(emb_dim = opt.emb_dim, voxel_size= opt.voxel_size)
    
    model= torch.nn.DataParallel(model)
    model.to(opt.device)
    
    if opt.load_model != '':
        model.load_state_dict(torch.load(opt.load_model))
    trainer = Trainer(opt.epochs, opt.device, opt.lr, opt.load_model, opt.result_dir, opt.batch_size, opt.num_views, opt.voxel_size, opt.log_dir, opt.emb_dim, opt.fusion)
    trainer.train(model, train_dataloader, eval_dataloader)
    
def test():
    opt = parser.parse_args() 
    opt.load_model = os.path.join(opt.result_dir,'models', 'cluster_views_15_emb_dim_1024_voxel_size_128_lr_0.001_epoch_2300.pth')
    train_dataset = BlenderDataset(mode = 'train', save_directory  = '/home/maturk/data/train_test_split/train', voxel_size=opt.voxel_size)
    eval_dataset = BlenderDataset(mode = 'train', save_directory  = '/home/maturk/data/train_test_split/test', voxel_size=opt.voxel_size)
    print('Current train and eval split: ', train_dataset.__len__(), eval_dataset.__len__()) 
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=1,
                                                   num_workers=opt.workers,
                                                   shuffle = False)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                   batch_size=1,
                                                   num_workers=opt.workers,
                                                   shuffle = False)
    
    model = VoxelAE(emb_dim = opt.emb_dim, voxel_size= opt.voxel_size).to(opt.device)
    if opt.load_model != '':
        model.load_state_dict(torch.load(opt.load_model))

    # Choose test examples
    obj_iter = iter(eval_dataloader)
    test_objects = []
    counter = 0
    for i in obj_iter:
        if counter%5 == 0:
            test_objects.append(i)
        counter +=1
        if counter>50:
            break
    print(len(test_objects))
    
    for i in range(counter):
        test_object = test_objects[i]
        class_dir =test_object['class_dir'].cpu().detach().numpy()[0]
        obj_dir = test_object['object_dir'][0]
        print(class_dir, obj_dir)
        
        gt_grid = test_object['gt_grid']
        depths = test_object['depths']
        xyzs = depths.clone().detach().to(device=opt.device, dtype= torch.float32)
        
        embedding, out_grid = model(xyzs)
        voxel_probs = out_grid.sigmoid() 
        
        #np.save(os.path.join('/home/maturk/git/ObjectReconstructor/ngp_pl/CustomData', f'{class_dir}_{obj_dir}_pred_grid.npy'), out_grid.cpu().detach().numpy())
        mse_loss = torch.nn.MSELoss()
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits
        out_mse_loss = mse_loss(voxel_probs, test_object['gt_grid'].to(opt.device))
        out_ce_loss = ce_loss(out_grid, test_object['gt_grid'].to(opt.device))
        print('MSE Loss: ', out_mse_loss, 'CE Loss: ', out_ce_loss)
        
        # Choose if you want to render and save predicted occupancy fields to a mesh and image file. 
        PYTORCH3D_VIS = True
        OBJECT_NAME = 'cluster'
        if PYTORCH3D_VIS == True:
            from PIL import Image
            images = pytorch3d_vis(gt_grid, view_elev=0)
            images = images[0, ..., :3].cpu().numpy() *255
            images = images.astype(np.uint8)
            im = Image.fromarray(images)
            im.save(os.path.join(opt.result_dir, 'images', f'gt_{OBJECT_NAME}_{class_dir}_{obj_dir}.png' ))
            # render a few different views based on elevation angle
            for elev in range(0,100,40):
                images = pytorch3d_vis(voxel_probs, view_elev=elev)
                images = images[0, ..., :3].cpu().numpy() *255
                images = images.astype(np.uint8)
                im = Image.fromarray(images)
                im.save(os.path.join(opt.result_dir, 'images', f'pred_{OBJECT_NAME}_{class_dir}_{obj_dir}_{elev}.png' )) 
                print('saved: ', elev)
            
def train_n_objects():
    N = 20
    opt = parser.parse_args() 
    opt.load_model = os.path.join(opt.result_dir, 'models', '20_models__voxels_views_15_emb_dim_1024_voxel_size_128_lr_0.001_epoch_3500.pth')
    opt.lr = 0.0001
    opt.epochs = 4501
    dataset = BlenderDataset(mode = 'train', save_directory  = '/home/maturk/data/multi_test', voxel_size=opt.voxel_size)
    train_dataset, eval_dataset, _ = random_split(dataset, [0.11, 0.89, 0.0], generator=torch.Generator().manual_seed(42))
    print('Current train and eval split: ', train_dataset.__len__(), eval_dataset.__len__()) 

    objects = []
    for n in range(N):
        objects.append(train_dataset.__getitem__(n))
    
    if opt.fusion:
        model = VoxelFusionAE(emb_dim = opt.emb_dim, voxel_size= opt.voxel_size).to(opt.device)
    else:
        model = VoxelAE(emb_dim = opt.emb_dim, voxel_size= opt.voxel_size).to(opt.device)
    
    if opt.load_model != '':
        model.load_state_dict(torch.load(opt.load_model))
        
    trainer = Trainer(opt.epochs, opt.device, opt.lr, opt.load_model, opt.result_dir, opt.batch_size, opt.num_views, opt.voxel_size, opt.log_dir, opt.emb_dim, opt.fusion)
    trainer.train_n_objects(model, objects)

def test_n_objects():
    N = 20
    opt = parser.parse_args() 
    dataset = BlenderDataset(mode = 'train', save_directory  = '/home/maturk/data/multi_test', voxel_size=opt.voxel_size)
    train_dataset, eval_dataset, _ = random_split(dataset, [0.11, 0.89, 0.0], generator=torch.Generator().manual_seed(42))
    print('Current train and eval split: ', train_dataset.__len__(), eval_dataset.__len__()) 
    objects = []
    for n in range(N):
        objects.append(train_dataset.__getitem__(n))
    
    if opt.fusion:
        model = VoxelFusionAE(emb_dim = opt.emb_dim, voxel_size= opt.voxel_size).to(opt.device)
    else:
        model = VoxelAE(emb_dim = opt.emb_dim, voxel_size= opt.voxel_size).to(opt.device)
        
    model.load_state_dict(torch.load(os.path.join(opt.result_dir, 'models', '20_models__voxels_views_15_emb_dim_1024_voxel_size_128_lr_0.0001_epoch_4500.pth')))
    
    CE_loss = torch.nn.functional.binary_cross_entropy_with_logits
    contrastive_loss = losses.contrastive_loss.ContrastiveLoss()
    
    batch_depths = []
    gt_pc = []
    classes = []
    directory = []
    for n in range(len(objects)):
        batch_depths.append(objects[n]['depths'].clone().detach().to(device=opt.device, dtype= torch.float32))
        gt_pc.append(objects[n]['gt_grid'].clone().detach().to(device=opt.device, dtype= torch.float32))
        classes.append(objects[n]['class_dir'])
        directory.append(objects[n]['object_dir'])
    
    batch_depths = torch.stack([depth for depth in batch_depths], dim = 0)
    gt_grids = torch.stack([gt for gt in gt_pc], dim = 0)
    embedding, grid_out = model(batch_depths)
    out_CE_loss = CE_loss(grid_out, gt_grids.to(opt.device))
    out_cont_loss = contrastive_loss(embedding, torch.Tensor(classes))
    voxel_probs = grid_out.sigmoid()
    
    out_loss = out_CE_loss + out_cont_loss
    print('shapes', batch_depths.shape, gt_grids.shape)
    print('CE loss: ', out_CE_loss, 'contrastive loss: ', out_cont_loss)
    
    # Choose if you want to render occupancy fields to images
    PYTORCH3D_VIS = False
    if PYTORCH3D_VIS == True:
        for n in range(len(batch_depths)):
            from PIL import Image
            images = pytorch3d_vis(gt_grids[n].unsqueeze(0), view_elev=0)
            print(f'Object number {n}, {int(classes[n])}, {str(directory[n])} rendering')
            images = images[0, ..., :3].cpu().numpy() *255
            images = images.astype(np.uint8)
            im = Image.fromarray(images)
            im.save(os.path.join(opt.result_dir, 'images', f'{N}_gt_{int(classes[n])}_{str(directory[n])}.png' ))
            
            # render a few different predicted images based on elevation
            for elev in range(0,80,80):
                images = pytorch3d_vis(voxel_probs[n].unsqueeze(0), view_elev=elev)
                images = images[0, ..., :3].cpu().numpy() *255
                images = images.astype(np.uint8)
                im = Image.fromarray(images)
                im.save(os.path.join(opt.result_dir, 'images', f'{N}_pred_{int(classes[n])}_{str(directory[n])}_{elev}.png' )) 

def test_test():
    N = 1
    opt = parser.parse_args() 
    dataset = BlenderDataset(mode = 'train', save_directory  = '/home/maturk/data/multi_test', voxel_size=opt.voxel_size)
    train_dataset, eval_dataset, _ = random_split(dataset, [0.11, 0.89, 0.0], generator=torch.Generator().manual_seed(42))
    print('Current train and eval split: ', train_dataset.__len__(), eval_dataset.__len__()) 
    objects = []
    objects.append(train_dataset.__getitem__(N))
    
    if opt.fusion:
        model = VoxelFusionAE(emb_dim = opt.emb_dim, voxel_size= opt.voxel_size).to(opt.device)
    else:
        model = VoxelAE(emb_dim = opt.emb_dim, voxel_size= opt.voxel_size).to(opt.device)
        
    model.load_state_dict(torch.load(os.path.join(opt.result_dir, 'models', '20_models__voxels_views_15_emb_dim_1024_voxel_size_128_lr_0.0001_epoch_4500.pth')))
    
    CE_loss = torch.nn.functional.binary_cross_entropy_with_logits
    contrastive_loss = losses.contrastive_loss.ContrastiveLoss()
    
    batch_depths = []
    gt_pc = []
    classes = []
    directory = []
    for n in range(len(objects)):
        batch_depths.append(objects[n]['depths'].clone().detach().to(device=opt.device, dtype= torch.float32))
        gt_pc.append(objects[n]['gt_grid'].clone().detach().to(device=opt.device, dtype= torch.float32))
        classes.append(objects[n]['class_dir'])
        directory.append(objects[n]['object_dir'])
    
    batch_depths = torch.stack([depth for depth in batch_depths], dim = 0)
    gt_grids = torch.stack([gt for gt in gt_pc], dim = 0)
    embedding, grid_out = model(batch_depths)
    out_CE_loss = CE_loss(grid_out, gt_grids.to(opt.device))
    out_cont_loss = contrastive_loss(embedding, torch.Tensor(classes))
    voxel_probs = grid_out.sigmoid()
    
    out_loss = out_CE_loss + out_cont_loss
    print('shapes', batch_depths.shape, gt_grids.shape)
    print('CE loss: ', out_CE_loss, 'contrastive loss: ', out_cont_loss)
    
    np.save(os.path.join('/home/maturk/git/ObjectReconstructor/ngp_pl/CustomData', 'pred_grid.npy'), grid_out.cpu().detach().numpy())
    
def render_grid():
    from PIL import Image
    opt = parser.parse_args()
    file = os.path.join('/home/maturk/git/ObjectReconstructor/ngp_pl/CustomData/bowl', 'optimised_grid.npy')
    grid = torch.Tensor(np.load(file)).to(device=opt.device).unsqueeze(0)
    pc = voxel_to_pc(grid.cpu())[0]
    
    import open3d as o3d
    print(pc.shape)
    print(pc)
    print(torch.count_nonzero(grid))
    print(grid.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.cpu().numpy())
    o3d.visualization.draw_geometries([pcd])
    for elev in range(0,80,20):
        print(elev)
        images = pytorch3d_vis(grid, view_elev=elev)
        images = images[0, ..., :3].cpu().numpy() *255
        images = images.astype(np.uint8)
        im = Image.fromarray(images)
        im.save(os.path.join(opt.result_dir, 'images', f'optimised_pred_{elev}.png' )) 
        print('saved ', elev)
    
if __name__ == "__main__":
    main()
    #test()
    #render_grid()
    