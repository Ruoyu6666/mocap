import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.timemil import TransLayer, initialize_weights, mexican_hat_wavelet
import random
from models.common import *
from models.inceptiontime import InceptionTimeFeatureExtractor
from models.nystrom_attention import NystromAttention




class WaveletEncoding(nn.Module):
    def __init__(self, dim=512, max_len = 256,hidden_len = 512,dropout=0.0):
        super().__init__()
       
        #n_w =3
        self.proj_1 = nn.Linear(dim, dim)
        self.proj_2 = nn.Linear(dim, dim)
        self.proj_3 = nn.Linear(dim, dim)
        

    def forward(self, x, wave1, wave2, wave3):
        x = x.transpose(1, 2)
        
        D = x.shape[1]
        scale1, shift1 =wave1[0,:],wave1[1,:]
        wavelet_kernel1 = mexican_hat_wavelet(size=(D,19), scale=scale1, shift=shift1)
        scale2, shift2 =wave2[0,:],wave2[1,:]
        wavelet_kernel2 = mexican_hat_wavelet(size=(D,19), scale=scale2, shift=shift2)
        scale3, shift3 =wave3[0,:],wave3[1,:]
        wavelet_kernel3 = mexican_hat_wavelet(size=(D,19), scale=scale3, shift=shift3)
        
         #Eq. 11
        pos1= torch.nn.functional.conv1d(x,wavelet_kernel1.unsqueeze(1),groups=D,padding ='same')
        pos2= torch.nn.functional.conv1d(x,wavelet_kernel2.unsqueeze(1),groups=D,padding ='same')
        pos3= torch.nn.functional.conv1d(x,wavelet_kernel3.unsqueeze(1),groups=D,padding ='same')
        x = x.transpose(1, 2)   #B*N*D

        #Eq. 10
        x = x + self.proj_1(pos1.transpose(1, 2) + pos2.transpose(1, 2) + pos3.transpose(1, 2))# + mixup_encording
        return x




class MIL(nn.Module):
    def __init__(self, in_features=512, mDim=64, n_classes=2, dropout=0., max_seq_len=400,
                 if_extract_feature=False, if_interval=False,instance_len=30):
        super().__init__()

        self.if_interval = if_interval
        self.if_extract_feature = if_extract_feature
        if if_interval:
            self.start_conv = StartConv(d_in=in_features, d_out=mDim)
            if self.if_extract_feature:
                self.feature_extractor = InceptionTimeFeatureExtractor(n_in_channels=mDim, out_channels=mDim // 4)
        else:
            if self.if_extract_feature:
                self.feature_extractor = InceptionTimeFeatureExtractor(n_in_channels=in_features, out_channels=mDim // 4)

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

        hidden_len = 2 * max_seq_len

        # No cls_token → positional encoding covers max_seq_len positions only
        self.pos_layer  = WaveletEncoding(mDim, max_seq_len, hidden_len)
        self.pos_layer2 = WaveletEncoding(mDim, max_seq_len, hidden_len)

        self.layer1 = TransLayer(dim=mDim, dropout=dropout)
        self.layer2 = TransLayer(dim=mDim, dropout=dropout)
        self.norm = nn.LayerNorm(mDim)
        self._fc2 = nn.Sequential(nn.Linear(mDim, mDim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(mDim, n_classes))
        self.alpha = nn.Parameter(torch.ones(1))

        initialize_weights(self)


    def forward(self, x, warmup=False):
        if self.if_interval:
            x = self.start_conv(x)
        if self.if_extract_feature:
            x = self.feature_extractor(x.transpose(1, 2))
            x = x.transpose(1, 2)

        # Keep a reference before transformer layers for warmup stabilisation
        global_token = x.mean(dim=1)           # (B, D) — raw mean before any encoding
        x = self.pos_layer(x,  self.wave1,  self.wave2,  self.wave3)   # WPE1
        x = self.layer1(x)                                               # TransLayer x1
        x = self.pos_layer2(x, self.wave1_, self.wave2_, self.wave3_)   # WPE2
        x = self.layer2(x)                                               # TransLayer x2
        x = self.norm(x)                     # (B, seq_len, D)

        x2 = x.mean(dim=1)                   # (B, D)  ← replaces cls_token extraction
        if warmup:
            x2 = self.alpha * x2 + (1-self.alpha) * global_token
        logits = self._fc2(x2)

        return x, logits
    