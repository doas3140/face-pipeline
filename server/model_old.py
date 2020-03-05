
from fastai import *
from fastai.vision import *

# SAME AS IN nbs/04_face_east_training.ipynb or facelib/face_east_training.py (same as .ipynb, just auto generated)

#export

# To work w/ ResNet18 change conv_layers in unet_concat:
# 3072 -> 768
# 640 -> 256
# 320 ->128
# (also pixelshuffle)

class EastModel(nn.Module):
    def __init__( self, num_bins, num_floats, pixelshuffle=False, conv_out=False, float_stats=None ):
        # float_stats.shape = [num_floats, 2], where 2 = (mean, std)
        super().__init__()
        self.conv_out = conv_out
        self.means = torch.tensor(imagenet_stats[0]).reshape(1,3,1,1) # [R,G,B]
        self.stds = torch.tensor(imagenet_stats[1]).reshape(1,3,1,1) # [R,G,B]
        
        if float_stats is not None:
            assert len(float_stats) == len(float_predictions)
            self.float_stats = True
            float_stats = torch.tensor(float_stats)
            self.float_means = float_stats[:,0].reshape(1,-1,1,1)
            self.float_stds = float_stats[:,1].reshape(1,-1,1,1)
        else: self.float_stats = False
        
        if pixelshuffle:
            unpool1 = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(512, 512*4, kernel_size=1))
            unpool2 = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(32,  32*4,  kernel_size=1))
            unpool3 = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(16,  16*4,  kernel_size=1))
        else:
            unpool1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            unpool2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            unpool3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            
#         self.resnet = models.resnet50(pretrained=True)
#         self.unet_block1 = UnetTypeConcat(unpool1, conv_layer(ni=3072, nf=128, ks=1))
#         self.unet_block2 = UnetTypeConcat(unpool2, conv_layer(ni=640,  nf=64,  ks=1))
#         self.unet_block3 = UnetTypeConcat(unpool3, conv_layer(ni=320,  nf=64,  ks=1))

        self.resnet = models.resnet18(pretrained=True) # (if using pixelshuffle u have to change those params too)
        self.unet_block1 = UnetTypeConcat(unpool1, conv_layer(ni=768, nf=128, ks=1))
        self.unet_block2 = UnetTypeConcat(unpool2, conv_layer(ni=256,  nf=64,  ks=1))
        self.unet_block3 = UnetTypeConcat(unpool3, conv_layer(ni=128,  nf=64,  ks=1))
        
        self.conv_bonus1 = conv_layer(ni=128,  nf=128, ks=3)
        self.conv_bonus2 = conv_layer(ni=64,   nf=64,  ks=3)
        self.conv1 = conv_layer(ni=64,   nf=32,  ks=3)
        self.conv2 = conv_layer(ni=32,   nf=32,  ks=3)
        self.conv_score = conv_layer(ni=32, nf=1,          ks=1, norm_type=None, use_activ=False)
        self.conv_geo   = conv_layer(ni=32, nf=4,          ks=1, norm_type=None, use_activ=False)
        self.conv_bin   = conv_layer(ni=32, nf=num_bins,   ks=1, norm_type=None, use_activ=False)
        self.conv_float = conv_layer(ni=32, nf=num_floats, ks=1, norm_type=None, use_activ=False)
#         self.conv_pts   = conv_layer(ni=32, nf=len(point_predictions),   ks=1, norm_type=None, use_activ=False)
#         self.conv_multi = conv_layer(ni=32, nf=len(binary_predictions),   ks=1, norm_type=None, use_activ=False)
#         self.conv10 = conv_layer(ni=32,  nf=1,   ks=1, norm_type=None, use_activ=False)
        
        if self.conv_out:
            self.conv_out = conv_layer(ni=32, nf=32, ks=3, stride=2, norm_type=None)
            
        # mask 
        self.unpool_mask = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
#         self.unpool_mask = F.interpolate(mask_pred, size=(h,w), mode='bilinear', align_corners=True)
        self.conv_mask = conv_layer(ni=32, nf=1, ks=1, norm_type=None, use_activ=False)
#         self.conv_mask = MaskHead(conv_layer(ni=32, nf=16, ks=1), conv_layer(ni=16, nf=3, ks=1), 
#                                   conv_layer(ni=3, nf=1, ks=1, norm_type=None, use_activ=False))
        

    def forward(self, x_orig):
        b,c,h,w = x_orig.shape
        x = x_orig
        x = (x - self.means.to(x.device)) / self.stds.to(x.device) # normalize
        
        f = []
        for layer in list(self.resnet.children())[:-1]:
            x = layer(x)
            if type(layer) == torch.nn.Sequential: f.append(x)
        
        # [b,128,h/16,w/16] = unet_block1([b,1024,h/16,w/16], [b,2048,h/32,w/32])
        o = self.unet_block1(f[2], f[3])
        o = self.conv_bonus1(o)
        # [b,64,h/8,w/8] = unet_block1([b,512,h/8,w/8], [b,128,h/16,w/16])
        o = self.unet_block2(f[1], o)
        o = self.conv_bonus2(o)
        # [b,64,h/4,w/4] = unet_block1([b,256,h/4,w/4], [b,64,h/8,w/8])
        o = self.unet_block3(f[0], o)

        o = self.conv1(o)  # [b,32,h/4,w/4]
        o = self.conv2(o)  # [b,32,h/4,w/4]
        
        if self.conv_out: o = self.conv_out(o) # [b,32,h/8,w/8]
        
        score = self.conv_score(o) # [b,1,h/4,w/4]
        geo_map = self.conv_geo(o) # [b,4,h/4,w/4]
        
        score = nn.Sigmoid()(score)
        geo_map = nn.Sigmoid()(geo_map)*2
        geo = geo_map
        
        bin_pred = nn.Sigmoid()( self.conv_bin(o) )
        float_pred = nn.Tanh()( self.conv_float(o) )*3 # [b,n,h,w]
        mask_pred = self.conv_mask(o)
        mask_pred = self.unpool_mask(mask_pred)
        mask_pred = nn.Sigmoid()( mask_pred.view(b,h,w) )
        
        if self.float_stats:
            float_pred = (float_pred * self.float_stds) + self.float_means
        
        other_lbls = [bin_pred, float_pred]
        
        return score, geo, other_lbls, mask_pred


class UnetTypeConcat(nn.Module):
    def __init__(self, unpool_layer, conv_layer):
        super().__init__()
        # conv_layer: has to ...
        # unpool_layer: has to increase [h,w] -> [h*2, w*2]
        self.unpool = unpool_layer
        self.conv = conv_layer
        
    def forward(self, e, d): # e/d - encoder/decoder output
        '''
        @param:  :encoder output
        @param:  :decoder output
        '''
        o = self.unpool(d)
        o = torch.cat((o, e), 1)
        return self.conv(o)