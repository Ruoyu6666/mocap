# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ELECTRA https://github.com/google-research/electra
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ["cls_token", "mask_token",]:
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("pos_embed"):
        return 0
    elif name.startswith("blocks"):
        return int(name.split(".")[1]) + 1
    else:
        return num_layers
