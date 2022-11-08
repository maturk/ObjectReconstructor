import argparse
import os
import random
from re import A
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from ObjectReconstructor.models.PSPNet import PSPNet
from torchvision.models.resnet import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms


psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()

        self.model = psp_models['resnet18'.lower()]()
        #self.cnn = resnet18(pretrained=True).eval()
        #self.model = create_feature_extractor(
        #    self.cnn, return_nodes={'layer4': 'layer4'})
        #self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                      std=[0.229, 0.224, 0.225]).cuda()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        #x = self.normalize(x)
        x = self.model(x)
        return x

class DenseFusionFeat(nn.Module):
    def __init__(self, num_points, emb_dim = 1024):
        super(DenseFusionFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, emb_dim, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1)    

        x = F.relu(self.conv5(pointfeat_2))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)
        #print('global feature is', ap_x, ap_x.shape)
        #ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        #return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1) #128 + 256 + 1024
        return x, ap_x

class DenseFusion(nn.Module):
    def __init__(self, num_points, emb_dim = 512, num_obj=1):
        super(DenseFusion, self).__init__()
        self.num_points = num_points
        self.emb_dim = emb_dim
        self.cnn = ModifiedResnet()
        self.feat = DenseFusionFeat(num_points, emb_dim=emb_dim)

    def forward(self, img, x, choose, obj):
        batch_x = torch.empty((x.shape[0], x.shape[1], self.emb_dim, self.num_points)).to(x.device)
        batch_ap_x = torch.empty((x.shape[0], x.shape[1], self.emb_dim, 1)).to(x.device)
        for batch_idx, (colors, points, mask) in enumerate(zip(img,x,choose)):
            out_img = self.cnn(colors)
            bs, di, _, _ = out_img.size()
            emb = out_img.view(bs, di, -1)
            choose = mask.unsqueeze(1).repeat(1, di, 1)
            emb = torch.gather(emb, 2, choose).contiguous()
            x = points.transpose(2, 1).contiguous()
            x, ap_x = self.feat(x, emb)
            batch_x[batch_idx][:]= x
            batch_ap_x[batch_idx][:]= ap_x
        
        avg_x = batch_x.mean(dim=1)
        avg_ap_x = batch_ap_x.mean(dim=1)
        return avg_x, avg_ap_x

if __name__ == "__main__":
    from ObjectReconstructor.datasets import BlenderDataset
    dataset = BlenderDataset(mode = 'train', save_directory  = '/home/maturk/data/test', num_points=500)
    test = dataset.__getitem__(4)
    dataloader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=1,
                                                   num_workers=1,
                                                   shuffle = True)
    model = DenseFusion(num_points = 500, num_obj = 1)
    model.cuda()
    for rep in range(1):
            for i, data in enumerate(dataloader):
                print(i) 
                colors = data['colors']
                depths = data['depths']
                choose = data['masks']
                gt_pc = data['gt_pc']
                
                img = Variable(colors[0]).float().cuda()
                img = img.permute(0,3,1,2)
                points = Variable(depths[0]).float().cuda()
                choose = Variable(choose[0]).unsqueeze(0).float().cuda()
                choose = choose.type(torch.int64)
                x, ap_x = model(img, points, choose, torch.Tensor([1]).long().cuda())

                print(x.shape, x)
                print(ap_x.shape, ap_x)
                if i == 0:
                    break
    #test()