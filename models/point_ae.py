'''
Point cloud auto-encoder 
Code adapted from CenterSnap: https://github.com/zubair-irshad/CenterSnap/blob/master/external/shape_pretraining/model/auto_encoder.py

Additions: 
* [x] Dense-fusion like encoder of color and depth channels (PoseNet). 
* [x] Support for multi-view and single-view fusion. 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from ObjectReconstructor.DenseFusion.lib.network import PoseNet 


class PointCloudEncoder(nn.Module):
    def __init__(self, emb_dim):
        self.emb_dim = emb_dim
        super(PointCloudEncoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(256, 256, 1)
        self.conv4 = nn.Conv1d(256, 1024, 1)
        self.fc = nn.Linear(1024, emb_dim)

    def forward(self, xyzs): 
        """
        Args:
            xyzs = (B, num_views, 3, N)
        """     
        batch_embeddings = torch.empty((xyzs.shape[0], xyzs.shape[1], self.emb_dim)).to(xyzs.device)
        for batch, xyz in enumerate(xyzs):
            nump = xyz.size()[2]
            x = F.relu(self.conv1(xyz))
            x = F.relu(self.conv2(x))
            global_feat = F.adaptive_max_pool1d(x, 1)
            x = torch.cat((x, global_feat.repeat(1, 1, nump)), dim=1)
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = torch.squeeze(F.adaptive_max_pool1d(x, 1), dim=2)
            embedding = self.fc(x)
            batch_embeddings[batch][:]= embedding
        
        avg_embeddings = batch_embeddings.mean(dim = 1)
        return avg_embeddings


class PointCloudDecoder(nn.Module):
    def __init__(self, emb_dim, n_pts):
        super(PointCloudDecoder, self).__init__()
        self.fc1 = nn.Linear(emb_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 3*n_pts)

    def forward(self, embedding):
        """
        Args:
            embedding: (B, 512)
        """
        bs = embedding.size()[0]
        out = F.relu(self.fc1(embedding))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out_pc = out.view(bs, -1, 3).float()
        return out_pc


class PointCloudAE(nn.Module):
    def __init__(self, emb_dim=512, num_points=1024):
        super(PointCloudAE, self).__init__()
        self.encoder = PointCloudEncoder(emb_dim)
        self.decoder = PointCloudDecoder(emb_dim, num_points)

    def forward(self, in_pcs, emb=None):
        """
        Args:
            in_pcs: (B, num_views, N, 3)
            emb: (B, 512)

        Returns:
            emb: (B, emb_dim)
            out_pc: (B, n_pts, 3)
        """
        if emb is None:
            xyzs = in_pcs.permute(0, 1, 3, 2)   
            emb = self.encoder(xyzs)
        out_pc = self.decoder(emb)
        return emb, out_pc


class PointFusionAE(nn.Module):
    def __init__(self, emb_dim=512, num_points = 1024):
        super(PointFusionAE, self).__init__()
        self.emb_dim = emb_dim
        self.num_points = num_points
        self.encoder = PoseNet(num_points = self.num_points, emb_dim = self.emb_dim)
        self.decoder = PointCloudDecoder(emb_dim=self.emb_dim, n_pts=self.num_points)
    
    def forward(self, colors, xyzs, masks, emb=None):
        """
        Args:
            in_pcs: (B, num_views, N, 3)
            emb: (B, 512)

        Returns:
            emb: (B, emb_dim)
            out_voxel: (B, voxel_size, voxel_size, voxel_size)
        """
        if emb is None:
            #xyzs = xyzs.permute(0, 1, 3, 2)
            colors = colors.permute(0,1,4,2,3)   
            x, ap_x = self.encoder(colors, xyzs, masks, obj = torch.Tensor([1]).long().cuda())
            ap_x = ap_x.squeeze(-1)
        out_pc = self.decoder(ap_x)
        return ap_x, out_pc


if __name__ == "__main__":
    pass