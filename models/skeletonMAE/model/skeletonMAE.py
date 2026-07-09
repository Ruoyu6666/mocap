import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings

from .layers import MLP, SkeleEmbed, Block, trunc_normal_, DropPath


class SkeletonMAE(nn.Module):
    def __init__(self, dim_in=3, dim_feat=256, decoder_dim_feat=256, depth=5, decoder_depth=5, 
                 num_heads=8, mlp_ratio=4, num_frames=120, num_joints=25, patch_size=1, t_patch_size=3,
                 qkv_bias=True, qk_scale=None, drop_rate=0., 
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, 
                 norm_skes_loss=False, dataset="mocap"): 
        
        super().__init__()
        self.dim_in = dim_in
        self.dim_feat = dim_feat

        self.num_frames = num_frames
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.t_patch_size = t_patch_size

        self.norm_skes_loss = norm_skes_loss
        self.dataset = dataset

        ##### MAE encoder specifics #####
        self.joints_embed = SkeleEmbed(dim_in, dim_feat, num_frames, num_joints, patch_size, t_patch_size)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        #self.dim_feat_list = [dim_feat * (2 ** i) for i in range(depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)])
        self.norm = norm_layer(dim_feat)

        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames//t_patch_size, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_joints//patch_size, dim_feat))
        trunc_normal_(self.temp_embed, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
#
        # self.proj_head = nn.Sequential(nn.Linear(dim_feat, dim_feat), nn.GELU(), nn.Linear(dim_feat, dim_feat))
        
        ##### MAE decoder specifics #####
        self.decoder_embed = nn.Linear(dim_feat, decoder_dim_feat, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim_feat))
        trunc_normal_(self.mask_token, std=.02)

        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_dim_feat)

        self.decoder_temp_embed = nn.Parameter(torch.zeros(1, num_frames//t_patch_size, 1, decoder_dim_feat))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 1, num_joints//patch_size, decoder_dim_feat))
        trunc_normal_(self.decoder_temp_embed, std=.02)
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
    """
    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # NM, TP * VP, dim 
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        noise[~self.valid_patch_mask] = 1.0 + noise[~self.valid_patch_mask] # force invalid patches to have high noise → always removed/masked
        # sort noise for each sample, ascend - small is keep, large is remove/mask
        ids_shuffle = torch.argsort(noise, dim=1)  
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) # [32, 199, 128]ss
        # generate the binary mask: 0 is keep input to encoder, 1 is remove/masked in input
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore) # unshuffle to get the binary mask
        mask[~self.valid_patch_mask] = 1.0
        ids_invalid = (~self.valid_patch_mask).nonzero(as_tuple=False) # each row is a (batch_idx, patch_idx)
        return x_masked, mask, ids_restore, ids_keep
    
    """
    # First mask entire frames may be more effective to learn temporal dynamics, then mask joints may be more effective to learn spatial correlations. 
    def random_masking(self, x, frame_mask_ratio=0.6, joint_mask_ratio=0.5):
        N, L, D = x.shape
        TP = self.joints_embed.t_grid_size
        VP = self.joints_embed.grid_size
        assert L == TP * VP, f"Expected L={TP * VP}, got {L}"
        device = x.device

        # 1. FRAME-LEVEL masking 
        frame_noise = torch.rand(N, TP, device=device)
        num_frames_keep = int(TP * (1 - frame_mask_ratio))
        frame_ids_shuffle = torch.argsort(frame_noise, dim=1)
        frame_ids_restore  = torch.argsort(frame_ids_shuffle, dim=1)

        frame_mask = torch.ones(N, TP, device=device)           # 1 = masked
        frame_mask[:, :num_frames_keep] = 0
        frame_mask = torch.gather(frame_mask, dim=1, index=frame_ids_restore)
        frame_mask_token = frame_mask.unsqueeze(-1).expand(N, TP, VP).reshape(N, L) # Expand to token space: [N, TP] → [N, TP*VP]

        # 2. JOINT-LEVEL masking
        joint_noise = torch.rand(N, L, device=device)
        force_remove = frame_mask_token.bool() | ~self.valid_patch_mask  # [N, T*V] Force already-frame-masked tokens and invalid tokens to high noise
        joint_noise[force_remove] = 2.0 + joint_noise[force_remove]     # push out of [0,1]

        ids_shuffle = torch.argsort(joint_noise, dim=1)         # [N, T*V]
        ids_restore  = torch.argsort(ids_shuffle, dim=1)

        num_valid_surviving = (~force_remove).sum(dim=1)                 # [N] tokens survive both masks
        num_keep = (num_valid_surviving.float() * (1 - joint_mask_ratio)).int()  # per-sample
        len_keep = num_keep.max().item()
        
        ids_keep = ids_shuffle[:, :len_keep]                    # Keep tokens: [N, len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(N, len_keep, D))                 # [N, len_keep, D]
        #  Build attention mask: True = valid token, False = padding (to be ignored by encoder)
        #encoder_attention_mask = torch.arange(len_keep, device=device).unsqueeze(0) < num_keep.unsqueeze(1)  # [N, len_keep]
        #x_masked = x_masked * encoder_attention_mask.unsqueeze(-1).float()
        
        # 3. Build final binary mask  (1 = should reconstruct)
        mask = torch.ones(N, L, device=device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)     # unshuffle → [N, T*V]
        mask[~self.valid_patch_mask] = 1.0
        #print(mask.sum())
        #print(mask.size()) 

        return x_masked, mask, ids_restore, ids_keep
    

    def forward_encoder(self, x, mask_ratio): # x: [NM, T, V, C]
        NM = x.shape[0]
        TP = self.joints_embed.t_grid_size
        VP = self.joints_embed.grid_size
        # Flag the valid patch
        data_mask  = (x != 0.0).all(dim=-1) #  [B, 300, J] -> True: 3 coordinates all exist
        patch_mask = data_mask.unfold(1, self.t_patch_size, self.t_patch_size)  # [B, 100, 10, t_patch_size]
        patch_mask = patch_mask.all(dim=-1) # [B, 100, 10]
        self.valid_patch_mask = patch_mask.reshape(NM,  TP * VP)  # [NM, 100 * J=1200]

        x = self.joints_embed(x) # embed skeletons NM, TP, VP, C
        x = x + self.pos_embed[:, :, :VP, :] + self.temp_embed[:, :TP, :, :]  # add pos & temp embed
        x = x.reshape(NM, TP * VP, -1)                               # x: [NM, 1200, 128]
        x, mask, ids_restore, _ = self.random_masking(x)#, mask_ratio) # masking: length -> length * mask_ratio:  [96, 119, 128], mask: [96, 1200]

        for idx, blk in enumerate(self.blocks):                      # apply Transformer blocks
            x = blk(x)
        x = self.norm(x)                                             # [NM, TP * VP * R, 128]

        """
        # reconstruct the full lengths of sequence
        latent_full = torch.zeros(NM, TP * VP, -1, device=x.device)
        latent_full[mask] = x.reshape(-1, self.dim_feat)
        proj_in = latent_full.view(NM, TP, VP, -1).mean(dim=2)       # (NM, TP, D)
        proj_in = proj_in.repeat_interleave(self.t_patch_size, dim=1)  # (NM, T, D)
        M = 1
        proj_in = proj_in.view(-1, M, TP * self.t_patch_size, self.dim_feat).mean(dim=1)              # (N, T, D)
        projection = self.proj_head(proj_in)                         # (N, T, D)
        """
        return x, mask, ids_restore #,projection



    def forward_decoder(self, x, ids_restore):
        NM = x.shape[0]
        TP = self.joints_embed.t_grid_size
        VP = self.joints_embed.grid_size

        x = self.decoder_embed(x) # embed tokens
        C = x.shape[-1]

        mask_tokens = self.mask_token.repeat(NM, TP * VP - x.shape[1], 1)       # append intra mask tokens to sequence
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)                                       # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])) # restore original ordering
        x = x_.view([NM, TP, VP, C])

        # add pos & temp embed
        x = x + self.decoder_pos_embed[:, :, :VP, :] + self.decoder_temp_embed[:, :TP, :, :]  # NM, TP, VP, C
        
        x = x.reshape(NM, TP * VP, C)
        for idx, blk in enumerate(self.decoder_blocks):                             # apply Transformer blocks
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)  # predictor projection

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
        x = x.reshape(shape=(NM, TP * VP, u * p * C))
        return x    # Output: x: (N, L, t_patch_size * patch_size * 3)


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
        reconstruct_mask = mask * self.valid_patch_mask.float()  # [NM, TP * VP]
        loss = (loss * reconstruct_mask).sum() / reconstruct_mask.sum().clamp(min=1.0)  # mean loss on removed valid joints
        
        return loss
    
    def forward(self, x, mask_ratio=0.80, **kwargs):

        if self.dataset == "mabe_mice":
            N, T, M, _ = x.shape 
            x = x.reshape(N, T, M, self.num_joints, self.dim_in)
        
        N, T, M, V, C = x.shape # (batch_size, T, num_individuals,  num_joints, 3)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, T, V, C) # [B, 300, 10, 3]

        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio) # latent: [3B, 119, 128], mask: [3B, 1200=300/t_patch_size*12],
        pred = self.forward_decoder(latent, ids_restore)                # [NM, TP * VP, C] = [3B, 1200, 2*3(t_patch_size)]
        # print(pred.shape, , mask.shape) [32, 1000, 9] , [32, 1000]
        loss = self.forward_loss(x, pred, mask)
        
        return loss, pred, mask