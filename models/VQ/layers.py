import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F




class ResidualLayer2d(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """
    def __init__(self, in_dim, h_dim, res_h_dim,
                 kernel_size=[], stride=[], padding=[]):
        super(ResidualLayer2d, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False)
        )
    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack2d(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers,
                 kernel_size=[], stride=[], padding=[]):
        super(ResidualStack2d, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList([ResidualLayer2d(in_dim, h_dim, res_h_dim)] * n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        return F.relu(x)




class ResidualLayer(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim,):
        super(ResidualLayer, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)



class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList([ResidualLayer(in_dim, h_dim, res_h_dim)] * n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        return F.relu(x)



# Kernel size? Stride? Padding? Compression factor?
class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta  maps to the latent space x -> z. For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim,
                 kernel_size=[[],[],[]], stride=[[],[],[]], padding=[[],[],[]],
                 compression_factor=12):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        
        if compression_factor == 2: # Conv1d, only use conv on temporal dimension
            self.conv_stack = nn.Sequential(
                nn.Conv1d(in_dim, h_dim // 2, kernel_size=kernel-1, stride=stride-1, padding=1),
                nn.ReLU(),
                nn.Conv1d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1),
                ResidualLayer(h_dim, h_dim, res_h_dim),
                )
        if compression_factor == 4:
            self.conv_stack = nn.Sequential(
                nn.Conv1d(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
                nn.ReLU(),
                nn.Conv1d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1),
                nn.ReLU(),
                nn.Conv1d(h_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
                ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
                )
        """
        if compression_factor == 3:
            self.spatial_encoder = nn.Sequential(
                nn.Linear(num_joints * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU())
            self.temporal_encoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        """
        # 2d Conv
        if compression_factor == 12:
            self.conv_stack = nn.Sequential(
                nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
                nn.ReLU(),
                nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1),
                nn.ReLU(),
                nn.Conv2d(h_dim, h_dim, kernel_size=kernel, stride=stride+1, padding=1),
                nn.ReLU(),
                nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
                ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers))
        if compression_factor == 24: # (1800 , 24) -> (600, 1)
            self.conv_stack = nn.Sequential(
                nn.Conv2d(in_dim, h_dim // 2, kernel_size=(3, 2), stride=(3, 2)),
                nn.ReLU(),
                nn.Conv2d(h_dim // 2, h_dim, kernel_size=(1, 3), stride=(1, 3)),
                nn.ReLU(),
                nn.Conv2d(h_dim, h_dim, kernel_size=(1, 4), stride=(1, 4)),
                ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers))
            
    def forward(self, x):
        """
        B, T, N, _ = x.shape
        x = x.reshape(B, T, -1)  # (B, T, N*2)
        
        spatial_feat = self.spatial_encoder(x)  # (B, T, hidden_dim)
        temporal_feat, _ = self.temporal_encoder(spatial_feat)  # (B, T, hidden_dim)
        """

        return self.conv_stack(x)



# Kernel size? Stride? Padding?
# Compression factor?
class Decoder(nn.Module):
    """
    The p_phi (x|z) network. Given a latent sample z p_phi maps back to the original space z -> x.
    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """
    def __init__(self, e_dim, h_dim, out_dim, # = in_dim of encoder 
                 n_res_layers, res_h_dim,
                 kernel_size=[], stride=[], padding=[],
                 compression_factor=12):
        super(Decoder, self).__init__()

        kernel = 4
        stride = 2

        if compression_factor == 2:
            self.inverse_conv_stack = nn.Sequential(
                    nn.ConvTranspose1d(e_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
                    ResidualLayer(h_dim, h_dim, res_h_dim),
                    nn.ReLU(),
                    nn.ConvTranspose1d(h_dim, out_dim, kernel_size=kernel, stride=stride, padding=1))    
        
        if compression_factor == 4:
            self.inverse_conv_stack = nn.Sequential(
                    nn.Conv1d(e_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
                    ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
                    nn.ConvTranspose1d(h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose1d(h_dim//2, out_dim, kernel_size=kernel, stride=stride, padding=1))
            
        if compression_factor == 12:
            self.inverse_conv_stack = nn.Sequential(
                nn.ConvTranspose2d(e_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
                ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
                nn.ConvTranspose2d(h_dim, h_dim, kernel_size=kernel+1, stride=stride+1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(h_dim // 2, out_dim, kernel_size=kernel, stride=stride, padding=1),)
        
        if compression_factor == 24:
            self.inverse_conv_stack = nn.Sequential(
                nn.ConvTranspose2d(e_dim, h_dim, kernel_size=(3, 2), stride=(3, 2)),
                ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
                nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=(1, 3), stride=(1, 3)),
                nn.ReLU(),
                nn.ConvTranspose2d(h_dim // 2, out_dim, kernel_size=(1, 4), stride=(1, 4)),)

    def forward(self, x):
        return self.inverse_conv_stack(x)




class VectorQuantizer(nn.Module):
   
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1/self.n_e, 1/self.n_e)
    
    def forward(self, z):
        z = z.permute(0, 2, 1).contiguous() # [32, 300, 64]
        z_flattened = z.view(-1, self.e_dim) # (B*L, C=e_dim) [9600, 64]
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
        
        # Encoding
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1) # [9600, 1]
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e, device=z.device) # num_frames, n_e
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.beta * e_latent_loss

        quantized = z + (quantized - z).detach()

        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        z_q = quantized.permute(0, 2, 1).contiguous()

        return loss, z_q, perplexity, min_encoding_indices







class VectorQuantizer2d(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.
    
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer2d, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1/self.n_e, 1/self.n_e)
    
    def forward(self, z): 
        """
        Map encoder output z to a discrete one-hot vector that is the index of the closest embedding e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline: 1. get encoder input (B,C,H,W)
                               2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten. z:[32, 450, 18, 64] compress factor==4
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim) # (B*H*W, C=e_dim) [32 * 8100, 64]
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
        
        # get the closest encoding
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e, device=z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        
        # compute loss
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2) # e_latent_loss + q_latent_loss
        # preserve gradients
        z_q = z + (z_q - z).detach()
        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous() # [B, 64, 600, 1]

        return loss, z_q, perplexity, min_encodings, min_encoding_indices#, self.embedding.weight,



