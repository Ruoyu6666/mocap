"""
SkeletonMAE with factorized (divided) space-time attention.

Key change vs. the original: instead of flattening (T, V) into one long sequence and running plain global self-attention over it, 
each transformer block now does SPATIAL attention (across joints, within a frame) first, then TEMPORAL attention (across frames, for a fixed joint).

Why the masking strategy also had to change: The original `random_masking` *removed* masked tokens and compacted the survivors into a shorter, 
irregular sequence (classic MAE token-dropping). After that compaction, a token's original (frame, joint) coordinates are scrambled and 
the per-sample kept-length isn't even constant — so there's no valid (T, V) grid left to reshape into for spatial/temporal attention.

To keep the (TP, VP) grid intact while still masking, this version switches to a BEiT/SimMIM-style scheme: masked positions are *substituted* with a
learned mask token rather than dropped, so every sample still has a full (TP, VP) grid all the way through the encoder and decoder. This is the
standard way to combine masking with divided/factorized attention. The trade-off is that the encoder now processes all TP*VP tokens instead of
only the visible ones (less compute-efficient than token-dropping MAE, but required for clean spatial-then-temporal factorization).
"""
import torch
import torch.nn as nn

import os
import sys
sys.path.append("/home/rguo_hpc/myfolder/mocap/") # Adds the current directory to the Python path
from models.skeletonMAE.model.layers import MLP, SkeleEmbed, Block, trunc_normal_, DropPath


class DividedSTBlock(nn.Module):
    """
    Factorized space-time transformer block.

    Given a token grid (N, T, V, C):
      1. SPATIAL sub-block: full pre-norm transformer block (attn + MLP) applied independently to each frame, attending across its V joints.
      2. TEMPORAL sub-block: full pre-norm transformer block applied independently to each joint, attending across its T frames.
    """
    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias, qk_scale,
                 drop, attn_drop, drop_path, norm_layer):
        super().__init__()
        self.spatial_block = Block(
            dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            norm_layer=norm_layer)
        self.temporal_block = Block(
            dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            norm_layer=norm_layer)

    def forward(self, x, T, V):
        # x: (N, T*V, C)
        N, L, C = x.shape
        assert L == T * V, f"expected T*V={T * V} tokens, got {L}"

        # ---- spatial attention: across joints, within each frame ----
        x = x.view(N, T, V, C).reshape(N * T, V, C)
        x = self.spatial_block(x)
        x = x.view(N, T, V, C)

        # ---- temporal attention: across frames, for each joint ----
        x = x.permute(0, 2, 1, 3).reshape(N * V, T, C)
        x = self.temporal_block(x)
        x = x.view(N, V, T, C).permute(0, 2, 1, 3).reshape(N, T * V, C)
        return x


class SkeletonMAE(nn.Module):
    def __init__(self, dim_in=3, dim_feat=256, decoder_dim_feat=256, depth=5, decoder_depth=5,
                 num_heads=8, mlp_ratio=4, num_frames=120, num_joints=25, patch_size=1, t_patch_size=3,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, norm_skes_loss=False, dataset="mocap"):

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
        self.joints_embed = SkeleEmbed(dim_in, dim_feat, num_frames, num_joints, patch_size, t_patch_size)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # CHANGED: divided space-time blocks instead of plain global-attention Blocks
        self.blocks = nn.ModuleList([
            DividedSTBlock(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(dim_feat)

        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames // t_patch_size, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_joints // patch_size, dim_feat))
        trunc_normal_(self.temp_embed, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        # CHANGED: this mask token now substitutes masked positions at the ENCODER input (grid-preserving masking), 
        # not re-inserted at the decoder as in the original token-dropping version
        self.mask_token_enc = nn.Parameter(torch.zeros(1, 1, 1, dim_feat))
        trunc_normal_(self.mask_token_enc, std=.02)

        ####### MAE decoder specifics #######
        self.decoder_embed = nn.Linear(dim_feat, decoder_dim_feat, bias=True)

        self.decoder_blocks = nn.ModuleList([
            DividedSTBlock(
                dim=decoder_dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_dim_feat)

        self.decoder_temp_embed = nn.Parameter(torch.zeros(1, num_frames // t_patch_size, 1, decoder_dim_feat))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 1, num_joints // patch_size, decoder_dim_feat))
        trunc_normal_(self.decoder_temp_embed, std=.02)
        trunc_normal_(self.decoder_pos_embed, std=.02)
        self.decoder_pred = nn.Linear(decoder_dim_feat, t_patch_size * patch_size * dim_in, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, frame_mask_ratio=0.4, joint_mask_ratio=0.5):
        """
        x: (N, TP, VP, C) — embedded, position-encoded token grid.

        Returns a boolean mask (N, TP, VP), True = masked (to reconstruct).
        NOTE: unlike the original, tokens are NOT dropped/compacted here —
        that would destroy the regular (TP, VP) grid that spatial/temporal
        factorized attention needs. Masked positions are substituted with
        a learned mask token afterwards (see forward_encoder), so the grid
        shape stays constant throughout.
        """
        N, TP, VP, C = x.shape
        device = x.device

        # 1. frame-level masking: some whole frames are dropped first
        frame_noise = torch.rand(N, TP, device=device)
        num_frames_keep = max(int(TP * (1 - frame_mask_ratio)), 1)
        frame_ids_shuffle = torch.argsort(frame_noise, dim=1)
        frame_mask = torch.ones(N, TP, device=device)  # 1 = masked
        frame_mask.scatter_(1, frame_ids_shuffle[:, :num_frames_keep], 0)
        frame_mask = frame_mask.bool()  # (N, TP)

        # 2. joint-level masking, applied independently per frame
        joint_noise = torch.rand(N, TP, VP, device=device)
        num_joints_keep = max(int(VP * (1 - joint_mask_ratio)), 1)
        joint_ids_shuffle = torch.argsort(joint_noise, dim=-1)
        joint_mask = torch.ones(N, TP, VP, device=device)  # 1 = masked
        joint_mask.scatter_(2, joint_ids_shuffle[:, :, :num_joints_keep], 0)
        joint_mask = joint_mask.bool()

        # masked if the whole frame was dropped OR the joint was individually masked
        mask = frame_mask.unsqueeze(-1).expand(N, TP, VP) | joint_mask
        return mask  # (N, TP, VP) bool

    def forward_encoder(self, x, mask_ratio=None):  # x: [NM, T, V, C]
        NM = x.shape[0]
        TP = self.joints_embed.t_grid_size
        VP = self.joints_embed.grid_size

        x = self.joints_embed(x)  # (NM, TP, VP, C)
        x = x + self.pos_embed[:, :, :VP, :] + self.temp_embed[:, :TP, :, :]

        mask = self.random_masking(x)  # (NM, TP, VP) bool
        w = mask.unsqueeze(-1).type_as(x)
        mask_tokens = self.mask_token_enc.expand(NM, TP, VP, -1)
        x = x * (1 - w) + mask_tokens * w  # substitute masked positions, grid stays full-size

        x = x.reshape(NM, TP * VP, -1)
        for blk in self.blocks:
            x = blk(x, TP, VP)  # spatial attention, then temporal attention
        x = self.norm(x)

        mask = mask.reshape(NM, TP * VP).float()  # 1 = masked/reconstruct, 0 = keep
        return x, mask

    def forward_decoder(self, x, TP, VP):
        NM = x.shape[0]
        x = self.decoder_embed(x)
        C = x.shape[-1]
        x = x.reshape(NM, TP, VP, C)
        x = x + self.decoder_pos_embed[:, :, :VP, :] + self.decoder_temp_embed[:, :TP, :, :]
        x = x.reshape(NM, TP * VP, C)

        for blk in self.decoder_blocks:
            x = blk(x, TP, VP)  # spatial attention, then temporal attention
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x

    def patchify(self, imgs):  # Input: imgs: (N, T, V, 3)
        NM, T, V, C = imgs.shape
        p = self.patch_size
        u = self.t_patch_size
        assert V % p == 0 and T % u == 0
        VP = V // p
        TP = T // u
        x = imgs.reshape(shape=(NM, TP, u, VP, p, C))
        x = torch.einsum("ntuvpc->ntvupc", x)
        x = x.reshape(shape=(NM, TP * VP, u * p * C))
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [NM, T, V, 3]
        pred: [NM, TP * VP, t_patch_size * patch_size * 3]
        mask: [NM, TP * VP], 1 = masked/reconstruct, 0 = keep
        """
        target = self.patchify(imgs)
        if self.norm_skes_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        reconstruct_mask = mask
        loss = (loss * reconstruct_mask).sum() / reconstruct_mask.sum().clamp(min=1.0)
        return loss

    def forward(self, x, mask_ratio=0.80, **kwargs):
        if self.dataset == "mabe_mice":
            N, T, M, _ = x.shape
            x = x.reshape(N, T, M, self.num_joints, self.dim_in)
        N, T, M, V, C = x.shape  # (batch_size, T, num_individuals, num_joints, 3)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, T, V, C)

        latent, mask = self.forward_encoder(x, mask_ratio)
        TP = self.joints_embed.t_grid_size
        VP = self.joints_embed.grid_size
        pred = self.forward_decoder(latent, TP, VP)
        loss = self.forward_loss(x, pred, mask)

        return loss, pred, mask