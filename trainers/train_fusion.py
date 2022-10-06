from xmlrpc.client import INVALID_XMLRPC
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ObjectReconstructor.utils import BlenderDataset
from ObjectReconstructor.models.models import PointCloudDecoder
from ObjectReconstructor.models.denseFusion import DenseFusion
import argparse
import numpy as np
#from pytorch3d.loss import chamfer_distance
from kaolin.metrics.pointcloud import chamfer_distance
from torch.utils.data import random_split
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument('--num_points', type=int, default=1024, help='number of points in point cloud')
parser.add_argument('--emb_dim', type=int, default=512, help='dimension of latent embedding')
parser.add_argument('--batch_size', type=int, default = 16, help='batch size')
parser.add_argument('--device', type=str, default='cuda:0', help='GPU to use')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=30, help='max number of epochs to train')
parser.add_argument('--load_model', type=str, default='', help='resume from saved model')
parser.add_argument('--result_dir', type=str, default='/home/maturk/git/ObjectReconstructor/results/ae', help='directory to save train results')
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

    def train(self, encoder, decoder, train_dataloader, eval_dataloader):
        optimizer = torch.optim.Adam([{'params': encoder.parameters()}, 
                                    {'params': decoder.parameters(), 'lr': 1e-3}
                                    ], lr=self.lr)
                    
        batch_idx = 0
        for epoch in range(self.epochs):
            encoder.train()
            for i, data in enumerate(train_dataloader):
                batch_depths = data['depths'][0].float().cuda()
                batch_colors = data['colors'][0].float().cuda()
                batch_colors = batch_colors.permute(0,3,1,2)
                batch_masks = data['masks'][0].unsqueeze(0).float().cuda()
                batch_masks = batch_masks.type(torch.int64)
                gt_pc = data['gt_pc'][0].float().cuda()

                # Encode embeddings and decode to point cloud
                x, ap_x = encoder(batch_colors, batch_depths, batch_masks, torch.Tensor([1]).long().cuda())
                ap_x = ap_x.squeeze(-1)

                pc_out = decoder(embedding = ap_x)
                loss = chamfer_distance(pc_out, data['gt_pc'].permute(0,2,1).to(self.device))
                loss.backward()
                optimizer.step() 

                batch_idx += 1
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx} Loss:{loss}")

            print(f"Epoch {epoch} finished, evaluating ... ") 
            encoder.eval()
            decoder.eval()
            val_loss = 0.0
            for i, data in enumerate(eval_dataloader, 1):
                batch_depths = torch.tensor(object['depths'][0]).float().unsqueeze(0).cuda()
                batch_colors = torch.tensor(object['colors'][0]).float().unsqueeze(0).cuda()
                batch_colors = batch_colors.permute(0,3,1,2)
                batch_masks = torch.tensor(object['masks'][0]).unsqueeze(0).float().cuda()
                batch_masks = batch_masks.type(torch.int64)
                gt_pc = torch.tensor(object['gt_pc']).float().unsqueeze(0).cuda()

                x, ap_x = encoder(batch_colors, batch_depths, batch_masks, torch.Tensor([1]).long().cuda())
                ap_x = ap_x.squeeze(-1)
                pc_out = decoder(embedding = ap_x)

                loss = chamfer_distance(pc_out, gt_pc.permute(0,2,1).to(self.device))
                val_loss += loss.item()
                if i % 20 == 0:
                    print(f"Batch {i} Loss: {loss}")
            val_loss = val_loss / i
            print(f"Epoch {epoch} test average loss: {val_loss}")
            print(f">>>>>>>>----------Epoch {epoch} eval test finish---------<<<<<<<<")
            # save model after each epoch
            if epoch % 100 == 0:
                torch.save(encoder.state_dict(), f"{self.results_dir}/full_fusion_encoder_{self.num_views}_{self.num_points}_{epoch}.pth")
                torch.save(decoder.state_dict(), f"{self.results_dir}/full_fusion_decoder_{self.num_views}_{self.num_points}_{epoch}.pth")
            
    def train_one_object(self, encoder, decoder, object):
        optimizer = torch.optim.Adam([{'params': encoder.parameters()}, 
                                    {'params': decoder.parameters(), 'lr': 1e-3}
                                    ], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.9)
        batch_idx = 0
        for epoch in range(self.epochs):
            encoder.train()
            batch_depths = torch.tensor(object['depths'][0]).float().unsqueeze(0).cuda()
            batch_colors = torch.tensor(object['colors'][0]).float().unsqueeze(0).cuda()
            batch_colors = batch_colors.permute(0,3,1,2)
            batch_masks = torch.tensor(object['masks'][0]).unsqueeze(0).float().cuda()
            batch_masks = batch_masks.type(torch.int64)
            gt_pc = torch.tensor(object['gt_pc']).float().unsqueeze(0).cuda()

            # Encode embeddings and decode to point cloud
            x, ap_x = encoder(batch_colors, batch_depths, batch_masks, torch.Tensor([1]).long().cuda())
            ap_x = ap_x.squeeze(-1)
            pc_out = decoder(embedding = ap_x)

            loss = chamfer_distance(pc_out, gt_pc.permute(0,2,1).to(self.device))
            loss.backward()
            optimizer.step() 
            scheduler.step()
            
            print(f"Epoch {epoch} finished, evaluating ... ") 
            encoder.eval()
            decoder.eval()
            val_loss = 0.0
            # Encoder decode
            x, ap_x = encoder(batch_colors, batch_depths, batch_masks, torch.Tensor([1]).long().cuda())
            ap_x = ap_x.squeeze(-1)
            pc_out = decoder(embedding = ap_x)
            loss = chamfer_distance(pc_out, gt_pc.permute(0,2,1).to(self.device))
            val_loss += loss.item()
            print(f"Epoch {epoch} test average loss: {val_loss}")
            print(f">>>>>>>>----------Epoch {epoch} eval test finish---------<<<<<<<<")
            # save model after each epoch 
            if epoch % 100 == 0:
                torch.save(encoder.state_dict(), f"{self.results_dir}/single_fusion_encoder_{self.num_views}_{self.num_points}_{epoch}.pth")
                torch.save(decoder.state_dict(), f"{self.results_dir}/single_fusion_decoder_{self.num_views}_{self.num_points}_{epoch}.pth")


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
    encoder = DenseFusion(num_points = opt.num_points).to(opt.device)
    decoder = PointCloudDecoder(emb_dim=opt.emb_dim, n_pts=opt.num_points).to(opt.device)

    trainer = Trainer(opt.epochs, opt.device, opt.lr, opt.load_model, opt.result_dir, opt.batch_size, opt.num_views, opt.num_points)
    trainer.train(encoder, decoder, train_dataloader, eval_dataloader)

def test():
    import os
    import copy
    opt = parser.parse_args() 
    encoder = DenseFusion(num_points = opt.num_points).to(opt.device)
    decoder = PointCloudDecoder(emb_dim=opt.emb_dim, n_pts=opt.num_points).to(opt.device)
    encoder.load_state_dict(torch.load(os.path.join(opt.result_dir,'fusion_encoder_16_1024_600.pth')))
    decoder.load_state_dict(torch.load(os.path.join(opt.result_dir,'fusion_decoder_16_1024_600.pth')))

    dataset = BlenderDataset(mode = 'train', save_directory  = '/home/maturk/data/test', num_points=opt.num_points)
    object = dataset.__getitem__(50)
    depths = torch.tensor(object['depths'][0]).float().unsqueeze(0).cuda()
    colors = torch.tensor(object['colors'][0]).float().unsqueeze(0).cuda()
    colors = colors.permute(0,3,1,2)
    masks = torch.tensor(object['masks'][0]).unsqueeze(0).float().cuda()
    masks = masks.type(torch.int64)
    gt_pc = torch.tensor(object['gt_pc']).float().unsqueeze(0).cuda()
    
    x, ap_x = encoder(colors, depths, masks, torch.Tensor([1]).long().cuda())
    ap_x = ap_x.squeeze(-1)
    
    pc_out = decoder(embedding = ap_x)
    pc_out = pc_out.squeeze(0).cpu().detach().numpy()

    pc_gt = object['gt_pc']
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

def train_one_object():
    opt = parser.parse_args() 
    opt.epochs = 700
    dataset = BlenderDataset(mode = 'train', save_directory  = '/home/maturk/data/test', num_points=opt.num_points)
    object = dataset.__getitem__(100)

    encoder = DenseFusion(num_points = opt.num_points).to(opt.device)
    decoder = PointCloudDecoder(emb_dim=opt.emb_dim, n_pts=opt.num_points).to(opt.device)

    trainer = Trainer(opt.epochs, opt.device, opt.lr, opt.load_model, opt.result_dir, opt.batch_size, opt.num_views, opt.num_points)
    trainer.train_one_object(encoder, decoder, object=object)

    depths = object['depths'][0]
    xyzs = torch.tensor(depths).to(device=opt.device, dtype= torch.float)
    xyzs = xyzs.unsqueeze(0)
    with torch.cuda.amp.autocast(enabled=True):
        embedding, pc_out = decoder(xyzs)
    pc_out = pc_out.squeeze(0).cpu().detach().numpy()
    pc_gt = object['gt_pc'].squeeze(0).cpu().detach().numpy()
    pcd_gt = o3d.geometry.PointCloud()
    pcd_out = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(pc_gt.transpose())
    pcd_out.points = o3d.utility.Vector3dVector(pc_out)
    pcd_gt.colors = o3d.utility.Vector3dVector()
    o3d.visualization.draw_geometries([pcd_out])

if __name__ == "__main__":
    main()
    #train_one_object()
    #test()