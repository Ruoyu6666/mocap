import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings

from .layers import trunc_normal_
from .layers_rope import SkeleEmbed, RotaryTemporalEmbedding, Block


class SkeletonMAE(nn.Module):
    def __init__(self, dim_in=3, dim_feat=256, decoder_dim_feat=256, depth=5, decoder_depth=5, 
                 num_heads=8, mlp_ratio=4, num_frames=120, num_joints=25, patch_size=1, t_patch_size=3,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., rope_ratio=0.5,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, norm_skes_loss=False, dataset="mocap"): 
        
        super().__init__()
        self.dim_in = dim_in
        self.dim_feat = dim_feat
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.t_patch_size = t_patch_size

        self.norm_skes_loss = norm_skes_loss
        self.dataset = dataset

        ####### MAE encoder specifics #######
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

        self.rope = RotaryTemporalEmbedding(dim=self.blocks[0].attn.rot_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_joints//patch_size, dim_feat))
        trunc_normal_(self.pos_embed, std=.02)
        
        ####### MAE decoder specifics #######
        self.decoder_embed = nn.Linear(dim_feat, decoder_dim_feat, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim_feat))
        trunc_normal_(self.mask_token, std=.02)

        self.decoder_blocks = nn.ModuleList([
            Block(dim=decoder_dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                rope_ratio=rope_ratio, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_dim_feat)
        self.decoder_rope = RotaryTemporalEmbedding(dim=self.decoder_blocks[0].attn.rot_dim)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 1, num_joints//patch_size, decoder_dim_feat))
        trunc_normal_(self.decoder_pos_embed, std=.02)

        self.decoder_pred = nn.Linear(decoder_dim_feat, t_patch_size * patch_size * dim_in, bias=True) # decoder to patch
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


    def random_masking(self, x, TP, VP, segment_mask_ratio=0.5, seg_len=5, joint_mask_ratio=0.5):
        N, L, D = x.shape
        assert L == TP * VP, f"Expected L={TP * VP}, got {L}"
        device = x.device

        # 1. SEGMENT-LEVEL frame masking: place a few length-`seg_len` segments per sample until roughly segment_mask_ratio of the TP frames are masked.
        num_frames_mask_target = max(int(TP * segment_mask_ratio), 1)
        num_segments = max(round(num_frames_mask_target / seg_len), 1)

        block_edges = torch.linspace(0, TP, num_segments + 1).long().tolist()
        frame_mask = torch.zeros(N, TP, dtype=torch.bool, device=device)  # True = masked
        for b in range(N):
            for s in range(num_segments):
                block_start, block_end = block_edges[s], block_edges[s + 1]
                block_size = block_end - block_start
                this_seg_len = min(seg_len, block_size)
                max_start = block_end - this_seg_len
                start = torch.randint(block_start, max_start + 1, (1,)).item()
                frame_mask[b, start:start + this_seg_len] = True
        frame_mask_token = frame_mask.unsqueeze(-1).expand(N, TP, VP).reshape(N, L)  # [N, T*V]
        """
        # 2. NO joint-level masking. Need a per-token ordering to compact the surviving tokens and later restore them in-place
        noise = torch.rand(N, L, device=device)
        noise[frame_mask_token] += 2.0  # guarantees masked tokens sort after kept ones
        ids_shuffle = torch.argsort(noise, dim=1)             # [N, T*V]
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        num_keep_per_sample = (~frame_mask_token).sum(dim=1)  # [N], varies slightly with overlap 
        """
        # 2. JOINT-LEVEL masking
        joint_noise = torch.rand(N, L, device=device)
        force_remove = frame_mask_token.bool()
        joint_noise[force_remove] = 2.0 + joint_noise[force_remove]      # push out of [0,1]
        ids_shuffle = torch.argsort(joint_noise, dim=1)     # [N, T*V]
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        num_valid_surviving = (~force_remove).sum(dim=1)    # [N] tokens survive both masks
        num_keep_per_sample = (num_valid_surviving.float() * (1 - joint_mask_ratio)).int()  # per-sample
        
        len_keep = max(num_keep_per_sample.max().item(), 1)
        ids_keep = ids_shuffle[:, :len_keep]                # Keep tokens: [N, len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(N, len_keep, D))

        # 3. Build final binary mask (1 = should be reconstructed)
        mask = torch.ones(N, L, device=device)
        mask[:, :len_keep] = 0       # 0 = keep (valid, not reconstructed), 1 = reconstruct
        mask = torch.gather(mask, dim=1, index=ids_restore) # unshuffle → [N, T*V]
        #print("mask", mask.sum(), mask.numel(), mask.sum()/mask.numel())
        return x_masked, mask, ids_restore, ids_keep
    
    
    def forward_encoder(self, x, downsample_rate=None, segment_mask_ratio=0.5, seg_len=5, joint_mask_ratio=0.5): # x: [NM, T, V, C]
        NM = x.shape[0]
        x = self.joints_embed(x)
        TP, VP = x.shape[1], x.shape[2]
        x = x + self.pos_embed[:, :, :VP, :] 
        x = x.reshape(NM, TP * VP, -1)

        x, mask, ids_restore, ids_keep = self.random_masking(x, TP, VP, segment_mask_ratio, seg_len, joint_mask_ratio)
        if downsample_rate is None:
            downsample_rate = torch.ones(NM, device=x.device)
        elif not torch.is_tensor(downsample_rate):
            downsample_rate = torch.full((NM,), float(downsample_rate), device=x.device)
        effective_stride = self.t_patch_size * downsample_rate  # [NM]
        t_idx_full = torch.arange(TP, device=x.device).unsqueeze(0) * effective_stride.unsqueeze(1)  # [NM, TP], real-time units
        t_idx_full = t_idx_full.unsqueeze(-1).expand(NM, TP, VP).reshape(NM, TP * VP)  
        t_idx = torch.gather(t_idx_full, dim=1, index=ids_keep)  # [NM, N_kept]
        cos, sin = self.rope(t_idx)

        for blk in self.blocks:
            x = blk(x, cos=cos, sin=sin)
        x = self.norm(x)                  # [NM, TP * VP * R, C]
        return x, mask, ids_restore, TP


    def forward_decoder(self, x, ids_restore, TP, downsample_rate=None):
        NM = x.shape[0]
        VP = self.joints_embed.grid_size
        x = self.decoder_embed(x)   # [NM, len_keep, C]
        C = x.shape[-1]

        mask_tokens = self.mask_token.repeat(NM, TP*VP - x.shape[1], 1) # append intra mask tokens to sequence
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C)) # restore original ordering
        x = x_.view([NM, TP, VP, C])
        x = x + self.decoder_pos_embed[:, :, :VP, :]
        x = x.reshape(NM, TP * VP, C)

        if downsample_rate is None:
            downsample_rate = torch.ones(NM, device=x.device)
        elif not torch.is_tensor(downsample_rate):
            downsample_rate = torch.full((NM,), float(downsample_rate), device=x.device)
        effective_stride = self.t_patch_size * downsample_rate  # [NM]
        t_idx = torch.arange(TP, device=x.device).unsqueeze(0) * effective_stride.unsqueeze(1)  # [NM, TP], real-time units
        t_idx = t_idx.unsqueeze(-1).expand(NM, TP, VP).reshape(NM, TP * VP)                       # [NM, TP*VP]
        cos, sin = self.decoder_rope(t_idx)

        for blk in self.decoder_blocks:
            x = blk(x, cos=cos, sin=sin)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x


    def patchify(self, imgs): # Input: imgs: (N, T, V, 3)
        NM, T, V, C = imgs.shape
        p = self.patch_size     # spatial patch size
        u = self.t_patch_size   # temporal patch size
        assert V % p == 0 and T % u == 0
        VP = V // p
        TP = T // u
        x = imgs.reshape(shape=(NM, TP, u, VP, p, C))
        x = torch.einsum("ntuvpc->ntvupc", x)
        return x.reshape(shape=(NM, TP * VP, u * p * C))    # (N, L, t_patch_size * patch_size * 3)


    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [NM, T, V, 3]
        pred: [NM, TP * VP, t_patch_size * patch_size * 3]
        mask: [NM, TP * VP], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)  # [NM, TP * VP, C]
        if self.norm_skes_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [NM, TP * VP], mean loss per patch
        reconstruct_mask = mask # * self.valid_patch_mask.float()  # [NM, TP * VP]
        loss = (loss * reconstruct_mask).sum() / reconstruct_mask.sum().clamp(min=1.0)  # mean loss on removed valid joints        
        return loss


    def forward(self, x, downsample_rate=None, 
                segment_mask_ratio=0.5, seg_len=5, joint_mask_ratio=0.5):
        if self.dataset == "mabe_mice":
            N, T, M, _ = x.shape 
            x = x.reshape(N, T, M, self.num_joints, self.dim_in)
        else:
            if x.ndim == 5:
                N, T, M, V, C = x.shape # (batch_size, T, num_individuals,  num_joints, 3)
                x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, T, V, C) # [B, 300, 10, 3]
            elif x.ndim == 4:
                N, T, V, C = x.shape
        latent, mask, ids_restore, TP = self.forward_encoder(x, downsample_rate, 
                                            segment_mask_ratio, seg_len, joint_mask_ratio)
        pred = self.forward_decoder(latent, ids_restore, TP, downsample_rate)   # [NM, TP * VP, C] = [3B, 1200, 2*3(t_patch_size)]
        loss = self.forward_loss(x, pred, mask) 
        return loss, pred, mask 
        # pred.shape, mask.shape: [B, TP * VP, t_patch_size * patch_size * 3] , [B, TP * VP]