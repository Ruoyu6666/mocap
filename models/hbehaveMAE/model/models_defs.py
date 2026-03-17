
from .general_hiera import GeneralizedHiera
from .hbehave_mae import HBehaveMAE


def gen_hiera(**kwdargs):
    return GeneralizedHiera(
        in_chans=1,
        embed_dim=kwdargs["init_embed_dim"],
        num_heads=kwdargs["init_num_heads"],
        patch_stride=kwdargs["patch_kernel"],
        patch_padding=(0, 0, 0),
        **kwdargs
    )


def hbehavemae(**kwdargs):
    return HBehaveMAE(
        in_chans=1,
        embed_dim=kwdargs["init_embed_dim"],
        num_heads=kwdargs["init_num_heads"],
        patch_stride=kwdargs["patch_kernel"],
        patch_padding=(0, 0, 0),
        **kwdargs
    )
