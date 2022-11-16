import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from ObjectReconstructor.DenseFusion.lib.network import PoseNet 
from ObjectReconstructor.DenseFusion.lib.network import PoseNet
from ObjectReconstructor.models.point_ae import PointCloudEncoder


class VoxelDecoder(nn.Module):
    def __init__(self, emb_dim = 512, voxel_size = 128):
        super(VoxelDecoder, self).__init__()
        self.voxel_size = voxel_size
        self.fc1 = nn.Linear(emb_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        
        # out_dimension = (Hin−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        if self.voxel_size == 64:
            self.deconv = nn.Sequential(
                    nn.ConvTranspose2d(1, 32, kernel_size=3, stride = 1, padding=1, output_padding= 0), 
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(32, 64, kernel_size=3, stride = 2, padding=1, output_padding= 1), 
                    nn.LeakyReLU(),
                    )
        if self.voxel_size == 128:
            self.deconv = nn.Sequential(
                    nn.ConvTranspose2d(1, 32, kernel_size=3, stride = 1, padding=1, output_padding= 0), 
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(32, 64, kernel_size=3, stride = 2, padding=1, output_padding= 1), 
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(64, 128, kernel_size=3, stride =2, padding=1, output_padding= 1),
                    nn.LeakyReLU()
                    )
        if self.voxel_size == 256:
            self.deconv = nn.Sequential(
                    nn.ConvTranspose2d(1, 32, kernel_size=3, stride = 1, padding=1, output_padding= 0), 
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(32, 64, kernel_size=3, stride = 2, padding=1, output_padding= 1), 
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(64, 128, kernel_size=3, stride =2, padding=1, output_padding= 1),
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(128, 256, kernel_size=3, stride =2, padding=1, output_padding= 1),
                    nn.LeakyReLU()
            )
            
        self.predictor = nn.Conv2d(self.voxel_size, self.voxel_size, kernel_size=1, stride=1, padding=0)
        
    def forward(self, embedding):
        """
        Args:
            embedding: (B, 512)
            out_voxel: (B, voxel_size, voxel_size, voxel_size)
        """
        bs = embedding.size()[0]
        out = F.relu(self.fc1(embedding))
        out = F.relu(self.fc2(out))
        out = out.view(bs, 1, 32, 32) # lifting from 1D to 2D
        for layer in self.deconv:
            out = layer(out)
        out_voxel = self.predictor(out)
        return out_voxel
    
    
class VoxelAE(nn.Module):
    def __init__(self, emb_dim=512, voxel_size=128):
        super(VoxelAE, self).__init__()
        self.encoder = PointCloudEncoder(emb_dim)
        self.decoder = VoxelDecoder(emb_dim, voxel_size)

    def forward(self, in_pcs, emb=None):
        """
        Args:
            in_pcs: (B, num_views, N, 3)
            emb: (B, 512)

        Returns:
            emb: (B, emb_dim)
            out_voxel: (B, voxel_size, voxel_size, voxel_size)
        """
        if emb is None:
            xyzs = in_pcs.permute(0, 1, 3, 2)   
            emb = self.encoder(xyzs)
        out_voxel = self.decoder(emb)
        return emb, out_voxel


class VoxelFusionAE(nn.Module):
    def __init__(self, emb_dim=512, voxel_size=128, num_points = 1024):
        super(VoxelFusionAE, self).__init__()
        self.emb_dim = emb_dim
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.encoder = PoseNet(num_points=1024,emb_dim=emb_dim)
        self.decoder = VoxelDecoder(emb_dim, voxel_size)

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
        out_voxel = self.decoder(ap_x)
        return ap_x, out_voxel