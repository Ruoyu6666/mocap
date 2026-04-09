# MAMP/model/transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from .layers import MLP, SkeleEmbed, Block, Attention, trunc_normal_, DropPath




class ActionHeadLinprobe(nn.Module):
    def __init__(self, dim_feat=512, num_classes=60, num_joints=25):
        super(ActionHeadLinprobe, self).__init__()
        self.fc = nn.Linear(dim_feat, num_classes)
        
    def forward(self, feat):
        N, M, T, J, C = feat.shape
        feat = feat.mean(dim=[1,2,3]) # [N, C]
        feat = self.fc(feat)
        return feat



class ActionHeadFinetune(nn.Module):
    def __init__(self, dropout_ratio=0., dim_feat=512, num_classes=60, num_joints=25, hidden_dim=2048):
        super(ActionHeadFinetune, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim_feat*num_joints, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, feat):
        N, M, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 1, 3, 4, 2)      # (N, M, T, J, C) -> (N, M, J, C, T)
        feat = feat.mean(dim=-1)
        feat = feat.reshape(N, M, -1)           # (N, M, J*C)
        feat = feat.mean(dim=1)                 # (N, J*C)
        feat = self.fc1(feat)                   # (N, hidden_dim)
        feat = self.bn(feat)
        feat = self.relu(feat)    
        feat = self.fc2(feat)
        return feat





class STTFEncoder(nn.Module):
    def __init__(self, dim_in=3, num_classes=3, dim_feat=256, depth=5, 
                 num_heads=8, mlp_ratio=4, num_frames=120, num_joints=25, patch_size=1, t_patch_size=4,
                 qkv_bias=True, qk_scale=None, drop_rate=0., 
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, 
                 protocol='linprobe'): # ["pretrain", "compute_representations","linprobe", "finetune"]
        
        super().__init__()

        self.num_classes = num_classes
        self.dim_feat = dim_feat

        self.num_frames = num_frames
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.t_patch_size = t_patch_size
        
        ##### MAE encoder specifics #####
        self.joints_embed = SkeleEmbed(dim_in, dim_feat, num_frames, num_joints, patch_size, t_patch_size)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, 
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)])
        self.norm = norm_layer(dim_feat)

        if protocol == 'linprobe':
            self.head = ActionHeadLinprobe(dim_feat=dim_feat, num_classes=num_classes)
        elif protocol == 'finetune':
            self.head = ActionHeadFinetune(dropout_ratio=0.3, dim_feat=dim_feat, num_classes=num_classes)
        """
        elif protocol == 'compute_representations': # only compute representaton no training
            self.head == None
        # maybe we can also add a protocol for linear probing with temporal pooling, 
        # i.e., pool the features across time and joints and then apply a linear classifier. 
        # This may be more effective for action recognition than the current linprobe protocol which applies linear classifier on each joint separately and then averages the predictions across joints. We can call this protocol 'linprobe_temporal_pooling' or something like that.
        else:
            raise TypeError('Unrecognized evaluation protocol!')
        """
        self.protocol = protocol

        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames//t_patch_size, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_joints//patch_size, dim_feat))
        trunc_normal_(self.temp_embed, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        # Initialize weights
        self.apply(self._init_weights)
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        #N, C, T, V, M = x.shape # [32, 300, 3, 12, 2]
        #x = x.permute(0, 4, 2, 3, 1).contiguous().view(N * M, T, V, C)
        
        # Modified for mabe dataset
        N, T, M, V, C = x.shape # for mabe dataset, M is number of mice. (batch_size, T, 3, V=12, C=2)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, T, V, C)
        
        
        x = self.joints_embed(x)    # embed skeletons
        NM, TP, VP, _ = x.shape
        x = x + self.pos_embed[:, :, :VP, :] + self.temp_embed[:, :TP, :, :] # add pos & temp embed
        x = x.reshape(NM, TP * VP, -1)

        for idx, blk in enumerate(self.blocks):
            x = blk(x)              # apply Transformer blocks
        
        x = self.norm(x)
        x = x.reshape(N, M, TP, VP, -1) # (B, 3, 100, 12, C)

        if self.protocol == "compute_representations":
            x = x.permute(0, 1, 2, 4, 3)  # (N, M, T, C, J)
            x = x.mean(dim=-1)            # (N, M, T, C) # calculate representation of each frame by averaging across joints
            x = x.mean(dim=1)             # (N, T, C)
        
        else:
            x = self.head(x)

        return x