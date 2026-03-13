import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from .layers import MLP, SkeleEmbed, Block, trunc_normal_, DropPath



class SkeletonMAE(nn.Module):
    def __init__(self, dim_in=3, dim_feat=256, decoder_dim_feat=256, depth=5, decoder_depth=5, 
                 num_heads=8, mlp_ratio=4,
                 num_frames=120, num_joints=25, patch_size=1, t_patch_size=4,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, 
                 norm_skes_loss=False):
        
        super().__init__()
        self.dim_feat = dim_feat

        self.num_frames = num_frames
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.t_patch_size = t_patch_size

        self.norm_skes_loss = norm_skes_loss
        
        ##### MAE encoder specifics #####
        self.joints_embed = SkeleEmbed(dim_in, dim_feat, num_frames, num_joints, patch_size, t_patch_size)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        #self.dim_feat_list = [dim_feat * (2 ** i) for i in range(depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, 
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)
            ])
        self.norm = norm_layer(dim_feat)

        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames//t_patch_size, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_joints//patch_size, dim_feat))
        trunc_normal_(self.temp_embed, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        ##### MAE decoder specifics #####
        self.decoder_embed = nn.Linear(dim_feat, decoder_dim_feat, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim_feat))
        trunc_normal_(self.mask_token, std=.02)

        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(decoder_depth)
            ])
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


    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random noise.
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device) # generate the binary mask: 0 is keep, 1 is remove
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep



    def forward_encoder(self, x, mask_ratio): # x: [3B, 300, 12, 2]
        
        x = self.joints_embed(x) # embed skeletons
        NM, TP, VP, _ = x.shape
        x = x + self.pos_embed[:, :, :VP, :] + self.temp_embed[:, :TP, :, :]  # add pos & temp embed
        x = x.reshape(NM, TP * VP, -1)                               # x: [96=3B, 1200, 128]
        x, mask, ids_restore, _ = self.random_masking(x, mask_ratio) # masking: length -> length * mask_ratio:  [96, 119, 128]
        # x: [24, 119, 128], mask: [24, 1200], ids_restore: [24, 100*12=1200]
        for idx, blk in enumerate(self.blocks):     # apply Transformer blocks
            x = blk(x)
        x = self.norm(x)
 
        return x, mask, ids_restore


    def forward_decoder(self, x, ids_restore):
        NM = x.shape[0]
        TP = self.joints_embed.t_grid_size
        VP = self.joints_embed.grid_size

        x = self.decoder_embed(x) # embed tokens
        C = x.shape[-1]

        # append intra mask tokens to sequence
        mask_tokens = self.mask_token.repeat(NM, TP * VP - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)                                       # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2]))# unshuffle
        x = x_.view([NM, TP, VP, C])

        # add pos & temp embed
        x = x + self.decoder_pos_embed[:, :, :VP, :] + self.decoder_temp_embed[:, :TP, :, :]  # NM, TP, VP, C
        
        # apply Transformer blocks
        x = x.reshape(NM, TP * VP, C)
        for idx, blk in enumerate(self.decoder_blocks):
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
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed joints

        return loss


    def forward(self, x, mask_ratio=0.80, **kwargs):
        # original version
        #N, C, T, V, M = x.shape
        #x = x.permute(0, 4, 2, 3, 1).contiguous().view(N * M, T, V, C)
        
        # Modified for mabe dataset
        N, T, M, V, C = x.shape # for mabe dataset, M is number of mice. (batch_size, T, 3, V=12, C=2)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, T, V, C)
        
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio) # latent: [3B, 119, 128], mask: [3B, 1200=300/t_patch_size*12],
        pred = self.forward_decoder(latent, ids_restore)                # [NM, TP * VP, C] = [3B, 1200, 2*3(downsasmpled patch size)]
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask




    