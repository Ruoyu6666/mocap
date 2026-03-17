import torch
import torch.nn as nn
import math
import warnings
import random
import numpy as np
from collections import OrderedDict
from functools import partial
from itertools import repeat
from drop import DropPath


class DSTformerMAE(nn.Module):
        def __init__(self, dim_in=3, dim_out=3, dim_feat=256, dim_rep=512,
                    depth=5, num_heads=8, mlp_ratio=4, decoder_depth = 3,
                    num_joints=17, maxlen=243, 
                    qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, att_fuse=True):
                super().__init__()

                self.dim_out = dim_out