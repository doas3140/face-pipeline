
import torch
import torch.nn as nn
from torchvision import models as torch_models


class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)


class FaceModel(nn.Module):
    def __init__(self, num_bins, num_floats, float_stats, num_fpts=68*2, d_model=512, poolsize=(8,8)):
        # float_stats.shape = [num_floats, 2], where 2 = (mean, std)
        super().__init__()
        self.means = imagenet_stats[0].reshape(1,3,1,1) # [R,G,B]
        self.stds = imagenet_stats[1].reshape(1,3,1,1) # [R,G,B]
        
        assert len(float_stats) == len(float_names)
        self.float_means = float_stats[:,0].reshape(1,-1)
        self.float_stds = float_stats[:,1].reshape(1,-1)

        resnet = torch_models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        h,w = poolsize
        self.res_head = nn.Sequential(nn.BatchNorm2d(512), nn.AdaptiveAvgPool2d([h,w]), Flatten(),
                                      nn.Linear(h*w*512, d_model), nn.ReLU(), 
                                      nn.BatchNorm1d(d_model), nn.Dropout(0.3))
                                   
        self.W_fpts = nn.Linear(d_model, num_fpts)
        self.W_bin = nn.Linear(d_model+num_fpts+num_floats, num_bins)
        self.W_float = nn.Linear(d_model+num_fpts, num_floats)
        

    def forward(self, x_orig):
        b,c,h,w = x_orig.shape
        x = x_orig
        x = (x - self.means.to(x.device)) / self.stds.to(x.device) # normalize
        
        o = self.resnet(x) # [b,512,h',w']
        o = self.res_head(o)
        
        fpts = nn.Tanh()(self.W_fpts(o))*2 # [b,68*2]
        floats = nn.Tanh()(self.W_float( torch.cat([o,fpts], dim=1) ))*3
        bins = nn.Sigmoid()(self.W_bin( torch.cat([o,fpts,floats], dim=1) ))
        
        
        floats = (floats * self.float_stds.to(floats.device)) + self.float_means.to(floats.device)
        return fpts.view(b,68,2), bins, floats