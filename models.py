import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PointCloudEncoder(nn.Module):
    def __init__(self, emb_dim):
        super(PointCloudEncoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(256, 256, 1)
        self.conv4 = nn.Conv1d(256, 1024, 1)
        self.fc = nn.Linear(1024, emb_dim)

    def forward(self, xyzs): 
        """
        Args:
            xyz: (B, 3, N)
            xyzs = (B, num_views, 3, N)
        """
        print('xyzs shape' , xyzs.shape)
        all_embeddings = torch.empty((xyzs.shape[0],1))
        for i, xyz in enumerate(xyzs):
            print(i)
            nump = xyzs.size()[2]
            x = F.relu(self.conv1(xyz))
            x = F.relu(self.conv2(x))
            global_feat = F.adaptive_max_pool1d(x, 1)
            x = torch.cat((x, global_feat.repeat(1, 1, nump)), dim=1)
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = torch.squeeze(F.adaptive_max_pool1d(x, 1), dim=2)
            embedding = self.fc(x)
            all_embeddings.append(embedding)
        
        avg_embedding = all_embeddings.mean()
        return avg_embedding


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
        out_pc = out.view(bs, -1, 3)
        return out_pc


class PointCloudAE(nn.Module):
    def __init__(self, emb_dim=512, n_pts=1024):
        super(PointCloudAE, self).__init__()
        self.encoder = PointCloudEncoder(emb_dim)
        self.decoder = PointCloudDecoder(emb_dim, n_pts)

    def forward(self, in_pcs, emb=None):
        """
        Args:
            in_pc: (B, N, 3)
            in_pcs: (B, num_views, N, 3)
            emb: (B, 512)

        Returns:
            emb: (B, emb_dim)
            out_pc: (B, n_pts, 3)

        """
        if emb is None:
            print('xyzs before', in_pcs.shape)
            xyzs = in_pcs.permute(0, 2, 1)
            print('xyzs after', xyzs.shape)
            emb = self.encoder(xyzs)
        out_pc = self.decoder(emb)
        return emb, out_pc