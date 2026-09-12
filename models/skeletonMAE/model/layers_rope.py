import torch
import torch.nn as nn

from .layers import MLP, DropPath


# num_frames not a hard constraint anymore
class SkeleEmbed(nn.Module):
    def __init__(self, dim_in=3, dim_feat=256, num_joints=25, patch_size=1, t_patch_size=3):
        super().__init__()
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.t_patch_size = t_patch_size
        self.grid_size = num_joints // patch_size
        kernel_size = [t_patch_size, patch_size]
        self.proj = nn.Conv2d(dim_in, dim_feat, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x):
        _, T, V, _ = x.shape
        assert V == self.num_joints, f"Input skeleton size ({V}) doesn't match model ({self.num_joints})."
        assert T % self.t_patch_size == 0, f"T ({T}) must be divisible by t_patch_size ({self.t_patch_size})."
        x = torch.einsum("ntsc->ncts", x)
        x = self.proj(x)                    # [N, dim_feat, T//t_patch_size, V//patch_size]  -- TP now dynamic
        x = torch.einsum("ncts->ntsc", x)   # [N, TP, VP, dim_feat]
        return x




class RotaryTemporalEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, t_idx):  # t_idx: [B, N] integer frame index per token
        freqs = torch.einsum("bn,d->bnd", t_idx.float(), self.inv_freq)  # [B, N, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)                          # [B, N, dim]
        return emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)            # [B, 1, N, dim]




def apply_rope(x, cos, sin):
    # x:        [B, heads, N, rot_dim] 
    # cos, sin: [B, 1,     N, rot_dim] (broadcast over heads)
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0., rope_ratio=0.5):
        
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.rot_dim = int(head_dim * rope_ratio) // 2 * 2  # keep even for chunking

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, valid_mask=None, seqlen=1, cos=None, sin=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if cos is not None and self.rot_dim > 0:
            q_rot, q_pass = q[..., :self.rot_dim], q[..., self.rot_dim:]
            k_rot, k_pass = k[..., :self.rot_dim], k[..., self.rot_dim:]
            q = torch.cat([apply_rope(q_rot, cos, sin), q_pass], dim=-1)
            k = torch.cat([apply_rope(k_rot, cos, sin), k_pass], dim=-1)

        if valid_mask is not None:
            k = k * valid_mask[:, None, :, None].to(k.dtype)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        if valid_mask is not None:
            key_mask = valid_mask[:, None, None, :].to(attn.dtype)
            attn = attn * key_mask
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x



class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., mlp_out_ratio=1., qkv_bias=True, 
                 qk_scale=None, drop=0., attn_drop=0., rope_ratio=0.5,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, rope_ratio)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_out_dim = int(dim * mlp_out_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=mlp_out_dim, act_layer=act_layer, drop=drop)

    
    def forward(self, x, valid_mask=None, seqlen=1, cos=None, sin=None):
        x = x + self.drop_path(self.attn(self.norm1(x), valid_mask, seqlen, cos=cos, sin=sin))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



def get_sinusoid_encoding(TP, dim, device):
    position = torch.arange(TP, device=device).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(TP, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [TP, dim]

"""
x = self.joints_embed(x)              # [NM, TP, VP, C]
TP, VP = x.shape[1], x.shape[2]
temp_embed = get_sinusoid_encoding(TP, self.dim_feat, x.device)   # [TP, C]
x = x + self.pos_embed[:, :, :VP, :] + temp_embed[None, :, None, :]
"""