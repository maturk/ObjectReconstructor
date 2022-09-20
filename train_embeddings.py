from asyncio.proactor_events import _ProactorBasePipeTransport
from random import shuffle
from sys import pycache_prefix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import BlenderDataset
from models import PointCloudEncoder, PointCloudAE
import argparse
import time
import numpy as np
#from pytorch3d.loss import chamfer_distance
from kaolin.metrics.pointcloud import chamfer_distance
from torch.utils.data import random_split
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument('--num_points', type=int, default=1028, help='number of points in point cloud')
parser.add_argument('--emb_dim', type=int, default=128, help='dimension of latent embedding')
parser.add_argument('--batch_size', type=int, default = 16, help='batch size')
parser.add_argument('--device', type=str, default='cuda:0', help='GPU to use')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=50, help='max number of epochs to train')
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

    def train(self, model, train_dataloader, eval_dataloader):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        batch_idx = 0
        for epoch in range(self.epochs):
            model.train()
            for i, data in enumerate(train_dataloader):
                xyzs = []
                batch_colors = data['colors']
                batch_depths = data['depths']
                for point_cloud in batch_depths:
                    xyzs.append(point_cloud)
                xyzs = torch.cat(xyzs).to(device = self.device , dtype = torch.float)
                xyzs = torch.reshape(xyzs, (self.batch_size, self.num_views - 1, self.num_points, 3))
                with torch.cuda.amp.autocast(enabled=True):
                    embedding, pc_out = model(xyzs)
                    loss= chamfer_distance(pc_out, data['gt_pc'].permute(0,2,1).to(self.device))
                loss.mean().backward()
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
                xyzs = xyzs.unsqueeze(0)
                with torch.cuda.amp.autocast(enabled=True):
                    embedding, pc_out = model(xyzs)
                    loss= chamfer_distance(pc_out, data['gt_pc'].permute(0,2,1).to(self.device))
                val_loss += loss.item()
                print(f"Batch {i} Loss: {loss}")
            val_loss = val_loss / i
            print(f"Epoch {epoch} test average loss: {val_loss}")
            print(f">>>>>>>>----------Epoch {epoch} eval test finish---------<<<<<<<<")
            # save model after each epoch
            torch.save(model.state_dict(), '{0}/model_{1:02d}.pth'.format(self.results_dir, epoch))

            

def main():
    opt = parser.parse_args() 
    dataset = BlenderDataset(mode = 'train', save_directory  = '/home/maturk/data/test', num_points=opt.num_points)
    test = dataset.__getitem__(4)
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
    encoder = PointCloudAE(emb_dim = opt.emb_dim, n_pts= opt.num_points).to(opt.device)
    if opt.load_model != '':
        encoder.load_state_dict(torch.load(opt.load_model))

    trainer = Trainer(opt.epochs, opt.device, opt.lr, opt.load_model, opt.result_dir, opt.batch_size, opt.num_views, opt.num_points)
    trainer.train(encoder, train_dataloader, eval_dataloader)


if __name__ == "__main__":
    main()