import random, sys 
import torch
import torch.nn as nn
import torch.nn.functional as F

# path_kd = '/home/avalocal/thesis23/KD'
# sys.path.append(path_kd)

import random
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.amp import autocast

# from dinoV2_dpt_models.backbone.dinov2 import DINOv2 #backbone
path = "/home/avalocal/thesis23/KD/Mask2Former_Multi_Task/mask2former/modeling/pixel_decoder"

sys.path.append(path)

from util.blocks import FeatureFusionBlock, _make_scratch
# from dinoV2_dpt_models.transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder

from typing import  Dict
import torch
from torch import nn


from detectron2.config import configurable
from detectron2.layers import  ShapeSpec
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False), 
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size, 
    )


class DPTHead(nn.Module):
    def __init__(
        self, 
        nclass,
        in_channels, 
        features=128, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024],
    ):
        super(DPTHead, self).__init__()      
        
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=True,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features, nclass, kernel_size=1, stride=1),
        )
        #1x1 conv layers before feeding to transformer decoder for adapting the features
        self.conv1 = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(features*2, features, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(features*4, features, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(features*8, features, kernel_size=1, stride=1, padding=0)
        self.lat_conv1 = nn.Conv2d(features, 256, kernel_size=1, stride=1, padding=0)
        self.lat_conv2 = nn.Conv2d(features, 256, kernel_size=1, stride=1, padding=0)
        self.lat_conv3 = nn.Conv2d(features, 256, kernel_size=1, stride=1, padding=0)
        self.lat_conv4 = nn.Conv2d(features, 256, kernel_size=1, stride=1, padding=0)
   
    
    
    def forward_features(self, out_features):

        # out = []
        '''<class 'dict'> this is the type of features
        4 this is the length of features
        dict_keys(['res2', 'res3', 'res4', 'res5']) these are the keys of features
        torch.Size([4, 96, 148, 296]) this is the shape of features[res2]
        torch.Size([4, 192, 74, 148]) this is the shape of features[res3]
        torch.Size([4, 384, 37, 74]) this is the shape of features[res4]
        torch.Size([4, 768, 19, 37]) this is the shape of features[res5]'''
  
        
        layer_1 = out_features['res2']
        layer_2 = out_features['res3']
        layer_3 = out_features['res4']
        layer_4 = out_features['res5']

        # layer_1, layer_2, layer_3, layer_4 = out_features

        # print(f"layer_1: {layer_1.shape}")
        # print(f"layer_2: {layer_2.shape}")
        # print(f"layer_3: {layer_3.shape}")
        # print(f"layer_4: {layer_4.shape}")
        # print("space\n")
        
        layer_1_rn = self.scratch.layer1_rn(layer_1) #torch.Size([4, 256, h//2, w//2])
        layer_2_rn = self.scratch.layer2_rn(layer_2) #torch.Size([4, 256, h//4, w//4])
        layer_3_rn = self.scratch.layer3_rn(layer_3) #torch.Size([4, 256, h//8, w//8])
        layer_4_rn = self.scratch.layer4_rn(layer_4) #torch.Size([4, 256, h//16, w//16])

        # print(f"layer_1_rn: {layer_1_rn.shape}")
        # print(f"layer_2_rn: {layer_2_rn.shape}")
        # print(f"layer_3_rn: {layer_3_rn.shape}")
        # print(f"layer_4_rn: {layer_4_rn.shape}")

        layer_1_rn = self.conv1(layer_1_rn)
        layer_2_rn = self.conv2(layer_2_rn)
        layer_3_rn = self.conv3(layer_3_rn)
        layer_4_rn = self.conv4(layer_4_rn)

        # #print num of parameters conv1, conv2, conv3, conv4
        # p1=sum(p.numel() for p in self.conv1.parameters())
        # p2=sum(p.numel() for p in self.conv2.parameters())
        # p3=sum(p.numel() for p in self.conv3.parameters())
        # p4=sum(p.numel() for p in self.conv4.parameters())
        # print(f"added parameters of convs: {p1+p2+p3+p4}") #0.98M
                                                            #2M w features384

        '''expand : True
        layer_1_rn: torch.Size([4, 256, 148, 296])
        layer_2_rn: torch.Size([4, 512, 74, 148])
        layer_3_rn: torch.Size([4, 1024, 37, 74])
        layer_4_rn: torch.Size([4, 2048, 19, 37])
        '''

        '''expand : False
        layer_1_rn: torch.Size([4, 256, 148, 296])
        layer_2_rn: torch.Size([4, 256, 74, 148])
        layer_3_rn: torch.Size([4, 256, 37, 74])
        layer_4_rn: torch.Size([4, 256, 19, 37])
         '''

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # print(f"path_4: {path_4.shape}")
        # print(f"path_3: {path_3.shape}")
        # print(f"path_2: {path_2.shape}")
        # print(f"path_1: {path_1.shape}")
        path_4 = self.lat_conv4(path_4)
        path_3 = self.lat_conv3(path_3)
        path_2 = self.lat_conv2(path_2)
        path_1 = self.lat_conv1(path_1)

        x = [
            path_4,     # lowest scale H/32, W/32
            path_3,     # 
            path_2,     # 
            # path_1,     # highest scale H, W
        ]
        # mask_features = self.conv1(path_1) 
        mask_features = path_1

        out_0 =  path_4

        return mask_features, out_0, x



@SEM_SEG_HEADS_REGISTRY.register()
class DPTPixelDecoder(nn.Module):
    @configurable
    def __init__(
        self, 
        input_shape: Dict[str, ShapeSpec],
        *,
        nclass=8,

        
    ):
        super().__init__()

        self.nclass = 8
        self.features = 384
        self.out_channels=[96, 192, 384, 768]
        self.use_bn = True
        self.embed_dim = 384
        self.head = DPTHead(self.nclass, self.embed_dim, self.features, self.use_bn, out_channels=self.out_channels)
        #self.head = Mask2formerHead(nclass, self.backbone.embed_dim, features, use_bn, out_channels=out_channels)


    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
       
        return ret
    # @autocast(device_type="cuda", enabled=False)
    def forward(self, x):
        return self.forward_features(x)   
    
    # @autocast(device_type="cuda", enabled=False)
    def forward_features(self, features):

        # patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        if self.head.__class__.__name__ == "DPTHead":
            mask_features,out_0, x = self.head.forward_features(features)
            # out = self.mask_decoder(x, mask_features, mask = None)
        
        # print(f"mask_features: {mask_features.shape}")
        # for i in range(len(x)):
        #     print(f"x[{i}]: {x[i].shape}")

        '''
        mask_features: torch.Size([4, 256, 296, 592])
        x[0]: torch.Size([4, 256, 37, 74])
        x[1]: torch.Size([4, 256, 74, 148])
        x[2]: torch.Size([4, 256, 148, 296])
        '''


        return mask_features, out_0, x
    #(out[-1]), out[0], multi_scale_features
    
