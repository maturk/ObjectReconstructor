import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ObjectReconstructor.utils import BlenderDataset
from ObjectReconstructor.models.models import PointCloudEncoder, PointCloudAE
import argparse
import time
import numpy as np
#from pytorch3d.loss import chamfer_distance
from kaolin.metrics.pointcloud import chamfer_distance
from torch.utils.data import random_split
import open3d as o3d
from pytorch_metric_learning import losses

parser = argparse.ArgumentParser()
parser.add_argument('--num_points', type=int, default=1024, help='number of points in point cloud')
parser.add_argument('--emb_dim', type=int, default=256, help='dimension of latent embedding')
parser.add_argument('--batch_size', type=int, default = 10, help='batch size')
parser.add_argument('--device', type=str, default='cuda:0', help='GPU to use')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=30, help='max number of epochs to train')
parser.add_argument('--load_model', type=str, default='', help='resume from saved model')
parser.add_argument('--result_dir', type=str, default='/home/maturk/git/ObjectReconstructor/results/ae', help='directory to save train results')
parser.add_argument('--save_dir', type=str, default='/home/maturk/data/test2', help='save directory of preprocessed shapenet images')
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir')
parser.add_argument('--workers', '-w', type=int, default=1)
parser.add_argument('--num_views', type=int, default=16, help = 'Number of input views per object instance')

class Trainer():
    def __init__(self,
                 epochs,
                 device,
                 lr,
                 load_model,
                 results_dir,
                 batch_size,
                 num_views,
                 num_points
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

    def train(self, model, train_dataloader, eval_dataloader):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        batch_idx = 0
        contrastive_loss = losses.contrastive_loss.ContrastiveLoss()

        for epoch in range(self.epochs):
            model.train()
            for i, data in enumerate(train_dataloader):
                optimizer.zero_grad()
                xyzs = []
                batch_depths = data['depths']
                for point_cloud in batch_depths:
                    xyzs.append(point_cloud)
                xyzs = torch.cat(xyzs).to(device = self.device , dtype = torch.float)
                batch_nums = int(xyzs.shape[0]/(self.num_views-1))
                xyzs = torch.reshape(xyzs, (batch_nums, self.num_views - 1, self.num_points, 3))

                with torch.cuda.amp.autocast(enabled=True):
                    embedding, pc_out = model(xyzs)
                    chamfer_loss = chamfer_distance(pc_out, data['gt_pc'].permute(0,2,1).to(self.device))
                    cont_loss = contrastive_loss(embedding,data['class_dir'])
                    loss = chamfer_loss.mean() + cont_loss
                loss.backward()
                optimizer.step() 

                batch_idx += 1
                if batch_idx % 100 == 0:
                    print(f"Batch {batch_idx} Loss:{loss}")

            print(f"Epoch {epoch} finished, evaluating ... ") 
            model.eval()
            val_loss = 0.0
            for i, data in enumerate(eval_dataloader, 1):
                xyzs = []
                batch_colors = data['colors']
                batch_depths = data['depths']
                for point_cloud in batch_depths:
                    xyzs.append(point_cloud)
                xyzs = torch.cat(xyzs).to(device = self.device , dtype = torch.float)
                batch_nums = int(xyzs.shape[0]/(self.num_views-1))
                xyzs = torch.reshape(xyzs, (batch_nums, self.num_views - 1, self.num_points, 3))
                with torch.cuda.amp.autocast(enabled=True):
                    embedding, pc_out = model(xyzs)
                    loss, _= chamfer_distance(pc_out, data['gt_pc'].permute(0,2,1).to(self.device))
                val_loss += loss.mean().item()
                if i % 20 == 0:
                    print(f"Batch {i} Loss: {loss}")
            val_loss = val_loss / i
            print(f"Epoch {epoch} test average loss: {val_loss}")
            print(f">>>>>>>>----------Epoch {epoch} eval test finish---------<<<<<<<<")
            # save model after each epoch
            torch.save(model.state_dict(), f"{self.results_dir}/{self.num_views}_{self.num_points}_{epoch}.pth")
    
    def train_one_object(self, model, object):
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.9)
            depths = object['depths']
            xyzs = torch.tensor(depths).to(device=self.device, dtype= torch.float32)
            xyzs = xyzs.unsqueeze(0)
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                embedding, pc_out = model(xyzs)
                loss = chamfer_distance(pc_out, object['gt_pc'].unsqueeze(0).permute(0,2,1).to(self.device))
                loss.backward()
                optimizer.step() 
                scheduler.step()
                
                # save model after each epoch
                if epoch % 250 == 0:
                    print('\n')
                    print('Epoch: ', epoch)
                    print('Loss: ', loss)
                    print('Embedding norm : ', embedding.norm(), 'Embedding mean: ', embedding.mean() )
                    print('Learning rate: ', scheduler.get_lr())
                    torch.save(model.state_dict(), f"{self.results_dir}/overfit_{self.num_views}_{self.num_points}_{epoch}.pth")
    
    def train_two_objects(self, model, object1, object2):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 1) 
        contrastive_loss = losses.contrastive_loss.ContrastiveLoss()
        batch_idx = 0
        
        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']
        
        batch_depths_1 = torch.tensor(object1['depths']).float().cuda()
        gt_pc_1 = torch.tensor(object1['gt_pc']).float().cuda()
        batch_depths_2 = torch.tensor(object2['depths']).float().cuda()
        gt_pc_2 = torch.tensor(object2['gt_pc']).float().cuda()
        
        batch_nums = int(2)
        
        batch_depths = torch.cat((batch_depths_1, batch_depths_2), dim = 0)
        gt_pc = torch.cat((gt_pc_1, gt_pc_2), dim = 0)
        
        batch_depths = batch_depths.view(batch_nums, self.num_views, self.num_points, 3)
        gt_pc = gt_pc.view(batch_nums, 3, 2048)
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            # Encode embeddings and decode to point cloud
            embedding, pc_out = model(batch_depths)
            chamfer_loss = chamfer_distance(pc_out, gt_pc.permute(0,2,1).to(self.device))
            cont_loss = contrastive_loss(embedding, torch.Tensor([object1['class_dir'], object2['class_dir']]))
            loss = cont_loss + chamfer_loss.mean()
            
            if epoch % 50 == 0 :
                print('\n')
                print('Epoch: ', epoch)
                print('chamfer loss: ', chamfer_loss, 'contrastive loss: ', cont_loss)
                print('norm 1 : ', embedding[0].norm(), 'norm 2 : ', embedding[1].norm())
                print('mean 1: ', embedding[0].mean(), 'mean 2: ', embedding[1].mean() )
                print('Total loss: ', loss)
                current_lr = get_lr(optimizer=optimizer)
                print('Current lr: ', current_lr)
            loss.backward()
            optimizer.step() 
            scheduler.step()
            
            ## save model after each epoch 
            if epoch % 200 == 0:
                torch.save(model.state_dict(), f"{self.results_dir}/double_PC_encoder_{self.num_views}_{self.num_points}_{epoch}.pth")


def main():
    opt = parser.parse_args() 
    dataset = BlenderDataset(mode = 'train', save_directory  = '/home/maturk/data/test', num_points=opt.num_points)
    train_split = int(0.80 * dataset.__len__())

    train_dataset, eval_dataset = random_split(dataset, [train_split, dataset.__len__()-train_split])
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=opt.batch_size,
                                                   num_workers=opt.workers,
                                                   shuffle = True)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                   batch_size=opt.batch_size,
                                                   num_workers=opt.workers,
                                                   shuffle = False)
    encoder = PointCloudAE(emb_dim = opt.emb_dim, num_pts= opt.num_points).to(opt.device)
    if opt.load_model != '':
        encoder.load_state_dict(torch.load(opt.load_model))

    trainer = Trainer(opt.epochs, opt.device, opt.lr, opt.load_model, opt.result_dir, opt.batch_size, opt.num_views, opt.num_points)
    trainer.train(encoder, train_dataloader, eval_dataloader)

def test():
    import os
    import copy
    opt = parser.parse_args() 
    model = PointCloudAE(emb_dim = opt.emb_dim, num_pts= opt.num_points).to(opt.device)
    model.load_state_dict(torch.load(os.path.join(opt.result_dir,'overfit_16_1028_1250.pth')))

    dataset = BlenderDataset(mode = 'train', save_directory  = '/home/maturk/data/test2', num_points=opt.num_points)
    test = dataset.__getitem__(0)
    depths = test['depths']
    xyzs = torch.tensor(depths).to(device=opt.device, dtype= torch.float)
    xyzs = xyzs.unsqueeze(0)
    with torch.cuda.amp.autocast(enabled=True):
        embedding, pc_out = model(xyzs)
    pc_out = pc_out.squeeze(0).cpu().detach().numpy()

    pc_gt = torch.Tensor.numpy(test['gt_pc']).transpose()
    pcd_gt = o3d.geometry.PointCloud()
    pcd_out = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(pc_gt)
    pcd_out.points = o3d.utility.Vector3dVector(pc_out)
    pcd_gt.colors = o3d.utility.Vector3dVector()
    o3d.visualization.draw_geometries([pcd_out, pcd_gt])

    def draw_registration_result(source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp],
                                        window_name='Point Cloud Registration',
                                        point_show_normal=False,
                                        mesh_show_wireframe=False,
                                        mesh_show_back_face=False)
    #draw_registration_result(pcd_out, pcd_gt, np.identity(4))

def train_one_object():
    opt = parser.parse_args() 
    opt.epochs = 1500
    opt.num_points = 1028 # 2048
    dataset = BlenderDataset(mode = 'train', save_directory  = '/home/maturk/data/test2', num_points=opt.num_points)
    object = dataset.__getitem__(0)
    encoder = PointCloudAE(emb_dim = opt.emb_dim, num_pts= opt.num_points).to(opt.device)
    trainer = Trainer(opt.epochs, opt.device, opt.lr, opt.load_model, opt.result_dir, opt.batch_size, opt.num_views, opt.num_points)
    trainer.train_one_object(encoder, object)

    depths = object['depths']
    xyzs = torch.tensor(depths).to(device=opt.device, dtype= torch.float)
    xyzs = xyzs.unsqueeze(0)
    with torch.cuda.amp.autocast(enabled=True):
        embedding, pc_out = encoder(xyzs)
    pc_out = pc_out.squeeze(0).cpu().detach().numpy()
    pc_gt = object['gt_pc'].squeeze(0).cpu().detach().numpy()
    pcd_gt = o3d.geometry.PointCloud()
    pcd_out = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(pc_gt.transpose())
    pcd_out.points = o3d.utility.Vector3dVector(pc_out)
    pcd_gt.colors = o3d.utility.Vector3dVector()
    o3d.visualization.draw_geometries([pcd_out])
    
def train_two_objects():
    import os
    opt = parser.parse_args() 
    opt.epochs = 1601
    dataset = BlenderDataset(mode = 'train', save_directory  = '/home/maturk/data/test2/', num_points=opt.num_points, num_views= opt.num_views)
    object1 = dataset.__getitem__(0)
    object2 = dataset.__getitem__(10)
    
    print('Class ids', object1['class_dir'], object2['class_dir'])
    model = PointCloudAE(emb_dim = opt.emb_dim, num_pts= opt.num_points).to(opt.device)

    LOAD_MODEL = False
    if LOAD_MODEL == True:
        model.load_state_dict(torch.load(os.path.join(opt.result_dir,'double_PC_encoder_16_1028_2000.pth')))

    trainer = Trainer(opt.epochs, opt.device, opt.lr, opt.load_model, opt.result_dir, opt.batch_size, opt.num_views, opt.num_points)
    trainer.train_two_objects(model, object1=object1, object2=object2)

def test_two_objects():
    import os
    import copy
    opt = parser.parse_args() 
    model = PointCloudAE( emb_dim=opt.emb_dim, num_pts = opt.num_points).to(opt.device)
    model.load_state_dict(torch.load(os.path.join(opt.result_dir,f'double_PC_encoder_{opt.num_views}_{opt.num_points}_600.pth')))

    dataset = BlenderDataset(mode = 'train', save_directory  = opt.save_dir, num_points=opt.num_points, num_views= opt.num_views)
    object = dataset.__getitem__(0)
    depths = torch.tensor(object['depths']).unsqueeze(0).float().cuda()
    colors = torch.tensor(object['colors'])
    colors = colors.permute(0,3,1,2).unsqueeze(0).float().cuda()
    masks = torch.tensor(object['masks']).unsqueeze(0).float().cuda()
    masks = masks.type(torch.int64)
    gt_pc = torch.tensor(object['gt_pc']).unsqueeze(0).float().cuda()
    
    with torch.cuda.amp.autocast(enabled=True):
        embedding, pc_out = model(depths)
        chamfer_loss = chamfer_distance(pc_out, gt_pc.permute(0,2,1))
    
    pc_out = pc_out.squeeze(0).cpu().detach().numpy()
    
    print(chamfer_loss)
    
    pc_gt = torch.Tensor.numpy(object['gt_pc'])
    pcd_gt = o3d.geometry.PointCloud()
    pcd_out = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(pc_gt.transpose())
    pcd_out.points = o3d.utility.Vector3dVector(pc_out)
    pcd_gt.colors = o3d.utility.Vector3dVector()
    o3d.visualization.draw_geometries([pcd_out])
    def draw_registration_result(source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp],
                                        window_name='Point Cloud Registration',
                                        point_show_normal=False,
                                        mesh_show_wireframe=False,
                                        mesh_show_back_face=False)
    draw_registration_result(pcd_out, pcd_gt, np.identity(4))

if __name__ == "__main__":
    #main()
    #test()
    #train_one_object()
    #train_two_objects()
    test_two_objects()
  