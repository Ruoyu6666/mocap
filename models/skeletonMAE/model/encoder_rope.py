import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import trunc_normal_
from .layers_rope import SkeleEmbed, Block, RotaryTemporalEmbedding


class STTFEncoder(nn.Module):
    def __init__(self, dim_in=3, num_classes=3, dim_feat=256, depth=5, 
                num_heads=8, mlp_ratio=4, num_frames=120, num_joints=25, patch_size=1, t_patch_size=3,
                qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., rope_ratio=0.5, 
                drop_path_rate=0., norm_layer=nn.LayerNorm, 
                protocol='compute_representations', dataset="mocap"):
        super().__init__()
        self.num_classes = num_classes

        self.dim_feat = dim_feat
        self.num_frames = num_frames    # reference/config only, no longer constrains forward
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.t_patch_size = t_patch_size
        
        self.joints_embed = SkeleEmbed(dim_in, dim_feat, num_joints, patch_size, t_patch_size)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, rope_ratio=rope_ratio, 
                drop_path=dpr[i], norm_layer=norm_layer) 
            for i in range(depth)])
        self.norm = norm_layer(dim_feat)

        # maybe also add a protocol for linear probing with temporal pooling, i.e., pool the features across time and joints 
        # and then apply a linear classifier. This may be more effective for action recognition than the linprobe protocol 
        # which applies linear classifier on each joint separately and then averages the predictions across joints. 
        # We can call this protocol 'linprobe_temporal_pooling' or something like that.
        
        self.protocol = protocol
        self.rope = RotaryTemporalEmbedding(dim=self.blocks[0].attn.rot_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_joints//patch_size, dim_feat))
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


    def forward(self, x, downsample_rate=None):
        if x.ndim == 5:
            N, T, M, V, C = x.shape
            x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, T, V, C)
        if x.ndim == 4:
            N, T, V, C = x.shape
            M =1
        NM = x.shape[0]

        data_mask  = (x != 0.0).all(dim=-1)
        patch_mask = data_mask.unfold(dimension=1, size=self.t_patch_size, step=self.t_patch_size)  # [B, 100, 10, t_patch_size]
        patch_mask = patch_mask.unfold(dimension=2, size=self.patch_size, step=self.patch_size)
        patch_mask = patch_mask.all(dim=-1).all(dim=-1) # [B, TP, VP]
        #self.valid_patch_mask = patch_mask.reshape(NM,  TP * VP)
        
        x = self.joints_embed(x) # [NM, TP, VP, C] -- TP dynamic, no fixed-length assert
        TP, VP = x.shape[1], x.shape[2]
        x = x + self.pos_embed[:, :, :VP, :]
        x = x.reshape(NM, TP * VP, -1)

        if downsample_rate is None:
            downsample_rate = torch.ones(NM, device=x.device)
        elif not torch.is_tensor(downsample_rate):
            downsample_rate = torch.full((NM,), float(downsample_rate), device=x.device)
        effective_stride = self.t_patch_size * downsample_rate  # [NM]
        t_idx = torch.arange(TP, device=x.device).unsqueeze(0) * effective_stride.unsqueeze(1)  # [NM, TP]
        t_idx = t_idx.unsqueeze(-1).expand(NM, TP, VP).reshape(NM, TP * VP)
        cos, sin = self.rope(t_idx)

        for blk in self.blocks:
            x = blk(x, cos=cos, sin=sin) #, self.valid_patch_mask)              # apply Transformer blocks
        x = self.norm(x)

        if self.protocol == "compute_representations":
            x = x.reshape(NM, TP, VP, -1)                        # [NM, TP, VP, C]
            joint_mask = patch_mask.unsqueeze(-1).float()        # [NM, TP, VP, 1]
            x = (x * joint_mask).sum(dim=2) / joint_mask.sum(dim=2).clamp(min=1) # joint-level masked mean (over VP) [NM, TP, C]
            x = x.reshape(N, M, TP, -1).mean(dim=1)              # [N, TP, C]
        else:
            x = x.reshape(N, M, TP, VP, -1)
            x = self.head(x)
        return x