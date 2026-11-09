import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .layers import Encoder, Decoder, VectorQuantizer



class VQVAE(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim,
                 n_e, e_dim, beta, 
                 kernel_size=[[],[],[]], stride=[[],[],[]], padding=[[],[],[]],
                 compression_factor=2):
        super(VQVAE, self).__init__()
        
        self.encoder = Encoder(in_dim, h_dim, n_res_layers, res_h_dim, kernel_size, stride, padding, compression_factor)
        self.pre_quant_conv = nn.Conv1d(h_dim, e_dim, kernel_size=1, stride=1)
        self.vq_layer = VectorQuantizer(n_e, e_dim, beta)
        self.decoder = Decoder(e_dim, h_dim, in_dim, n_res_layers, res_h_dim, kernel_size, stride, padding, compression_factor)

    def forward(self, x):
        z = self.encoder(x)
        z = self.pre_quant_conv(z)
        vq_loss, z_q, perplexity, min_encoding_indices = self.vq_layer(z)
        x_recon = self.decoder(z_q)

        return vq_loss, x_recon, perplexity, z_q, min_encoding_indices
    

    

class VQVAE2d(nn.Module):
    """
    VQ-VAE model that combines the encoder, vector quantizer, and decoder.
    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2

    - compression_factor : factor by which the input is downsampled
    """
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim,
                 n_e, e_dim, beta, 
                 kernel_size=[[],[],[]], stride=[[],[],[]], padding=[[],[],[]],
                 compression_factor=4):
        super(VQVAE2d, self).__init__()
        
        self.encoder = Encoder(in_dim, h_dim, n_res_layers, res_h_dim, kernel_size, stride, padding, compression_factor)
        self.projections =  nn.Linear(600, 1800) # intermediates result for represetation learning

        self.pre_quant_conv = nn.Conv2d(h_dim, e_dim, kernel_size=1, stride=1)
        self.vq_layer = VectorQuantizer(n_e, e_dim, beta)
        self.decoder = Decoder(e_dim, h_dim, in_dim, n_res_layers, res_h_dim, kernel_size, stride, padding, compression_factor)

    def forward(self, x, return_intermediates: bool = False): #x: [32, 3, 1800, 24]
        z = self.encoder(x) # [32, 128, 600, 1]
       
        z = self.pre_quant_conv(z) # [32, 64, 600, 1]
        vq_loss, z_q, perplexity, min_encodings, min_encoding_indices = self.vq_layer(z)
        
        intermediates = torch.squeeze(z_q, -1) # [32, 64, 600]
        intermediates = self.projections(intermediates) # [32, 128, 1800]
        
        x_recon = self.decoder(z_q)

        return vq_loss, x_recon, perplexity, min_encodings, min_encoding_indices, intermediates
    
