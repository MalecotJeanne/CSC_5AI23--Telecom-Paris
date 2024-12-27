import torch
import torch.nn as nn
import einops
import sys 
import os

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from models.blocks import EncoderBlock, DecoderBlock

class VAE(nn.Module):
    def __init__(self, config):
        super(VAE, self).__init__()

        self.batch_size = config["batch_size"]  
        self.n_channels = config["n_channels"]
        self.channels = config["channels"]

        self.r = len(self.channels) #reduction factor (nb of conv blocks)
        self.latent_dim = config["n_embbeding"]

        print(self.batch_size, self.n_channels, self.channels, self.r, self.latent)

        self.encoder = EncoderBlock(self.n_channels, self.channels)
        
        self.mu = nn.Linear(self.channels[-1]*(28//(2**self.r))**2, self.latent_dim)
        self.sigma = nn.Linear(self.channels[-1]*(28//(2**self.r))**2, self.latent_dim)
        self.decode_linear = nn.Linear(self.latent_dim, self.channels[-1]*(28//(2**self.r))**2)
        
        self.decoder = DecoderBlock(self.n_channels, self.channels)
        
        self.final_conv = nn.Conv2d(self.n_channels, self.n_channels,  kernel_size = 3, padding = "same")

    def _sampling(self, x):
        
        mu = self.mu(x)
        logvar = self.sigma(x)

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return mu + eps.mul(std), mu, logvar

    def forward(self, x):

        x = self.encoder(x)
        x = einops.rearrange(x, 'b c h w -> b (c h w)')
        
        x, mu, log_var = self._sampling(x)
        
        x = self.decode_linear(x)
        
        x = einops.rearrange(x, 'b (c h w) -> b c h w', h = 28//(2**self.r), w = 28//(2**self.r))

        x = self.decoder(x)

        x = self.final_conv(x)
        return x, mu, log_var   
        