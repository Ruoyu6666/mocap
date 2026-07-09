import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import random
from models.common import *
from models.inceptiontime import InceptionTimeFeatureExtractor
from models.nystrom_attention import NystromAttention


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            # m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dropout=0.2, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,  # number of landmarks
            pinv_iterations = 6,     # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=dropout
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x
    


### Define Wavelet Kernel
def mexican_hat_wavelet(size, scale, shift): #size :d*kernelsize  scale:d*1 shift:d*1
    """
    Generate a Mexican Hat wavelet kernel.

    Parameters:
    size (int): Size of the kernel.
    scale (float): Scale of the wavelet.
    shift (float): Shift of the wavelet.

    Returns:
    torch.Tensor: Mexican Hat wavelet kernel.
    """
    x = torch.linspace(-( size[1]-1)//2, ( size[1]-1)//2, size[1]).cuda()
    x = x.reshape(1,-1).repeat(size[0],1)
    x = x - shift  # Apply the shift

    # Mexican Hat wavelet formula
    C = 2 / ( 3**0.5 * torch.pi**0.25)
    wavelet = C * (1 - (x/scale)**2) * torch.exp(-(x/scale)**2 / 2)*1  /(torch.abs(scale)**0.5)

    return wavelet #d*L





class TimeMIL(nn.Module):
    def __init__(self, in_features=512, mDim=64, n_classes=2, dropout=0.,max_seq_len=400,
                 if_extract_feature=False, if_interval=False, instance_len=30 # length of each patch
                ):
        super().__init__()

        self.if_interval = if_interval                  # if downsize the time series by pooling in each patch
        self.if_extract_feature = if_extract_feature    # False: use pretrained feature
        if if_interval:
            self.start_conv=StartConv(d_in=in_features, d_out=mDim)
            if self.if_extract_feature:
                self.feature_extractor = InceptionTimeFeatureExtractor(n_in_channels=mDim, out_channels=mDim//4)
        else:
            if self.if_extract_feature:
                self.feature_extractor = InceptionTimeFeatureExtractor(n_in_channels=in_features, out_channels=mDim//4)# backbone can be replace here

        # Define WPE    
        self.wave1 = torch.randn(2, mDim, 1)
        self.wave1[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)
        self.wave1 = nn.Parameter(self.wave1)
        
        self.wave2 = torch.zeros(2, mDim, 1)
        self.wave2[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)
        self.wave2 = nn.Parameter(self.wave2)
        
        self.wave3 = torch.zeros(2, mDim, 1)
        self.wave3[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)
        self.wave3 = nn.Parameter(self.wave3)  
            
        self.wave1_ = torch.randn(2, mDim, 1)
        self.wave1_[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)
        self.wave1_ = nn.Parameter(self.wave1_)

        self.wave2_ = torch.zeros(2, mDim, 1)
        self.wave2_[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)
        self.wave2_ = nn.Parameter(self.wave2_)

        self.wave3_ = torch.zeros(2, mDim, 1)
        self.wave3_[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)
        self.wave3_ = nn.Parameter(self.wave3_)

        hidden_len = 2* max_seq_len
            
        # define class token      
        self.cls_token = nn.Parameter(torch.randn(1, 1, mDim))
        self.pos_layer =  WaveletEncoding(mDim, max_seq_len, hidden_len) 
        self.pos_layer2 = WaveletEncoding(mDim, max_seq_len, hidden_len) 
        # self.pos_layer = ConvPosEncoding1D(mDim)
        self.layer1 = TransLayer(dim=mDim, dropout=dropout)
        self.layer2 = TransLayer(dim=mDim ,dropout=dropout)
        self._fc2 = nn.Sequential(nn.Linear(mDim, mDim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(mDim, n_classes))
        self.alpha = nn.Parameter(torch.ones(1))
        
        initialize_weights(self)
        
        
    def forward(self, x, warmup=False):
        if self.if_interval:
            x = self.start_conv(x)
        if self.if_extract_feature:
            x = self.feature_extractor(x.transpose(1, 2))
            x = x.transpose(1, 2) # 3736, 1800, 128

        B, seq_len, D = x.shape
        global_token = x.mean(dim=1)#[0]
        # Create class token and concatenate with input
        cls_tokens = self.cls_token.expand(B, -1, -1)    #B * 1 * d
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_layer(x,self.wave1, self.wave2, self.wave3)   # WPE1
        x = self.layer1(x) # TransLayer x1
        x = self.pos_layer2(x,self.wave1_, self.wave2_, self.wave3_) # WPE2
        x = self.layer2(x) # TransLayer x2
        representation = x[:, 1:]
        x = x[:, 0]         # only cls_token is used for cls
        
        # stablity of training random initialized global token
        if warmup:
            x = self.alpha * x + (1-self.alpha) * global_token
        logits = self._fc2(x)
            
        return representation, logits


if __name__ == "__main__":
    x = torch.randn(3, 400, 4).cuda()
    model = TimeMIL(in_features=4,mDim=128).cuda()
    ylogits =model(x)
    print(ylogits.shape)

