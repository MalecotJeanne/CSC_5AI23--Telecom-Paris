import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import sys 
import os

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from models.blocks import EncoderBlock, DecoderBlock

class VectorQuantizer(nn.Module):
    def __init__(self, n_embedding, embedding_dim, beta = 0.5, device ='cpu'):
        super(VectorQuantizer, self).__init__()

        self.K = n_embedding
        self.embedding_dim = embedding_dim

        self.beta = beta
        
        self.embedding = nn.Embedding(self.K, self.embedding_dim).to(device)
        self.embedding.weight.data.uniform_(-1/n_embedding, 1/n_embedding) #https://huggingface.co/blog/ariG23498/understand-vq

    def forward(self, x):

        _ , _ , height, weight = x.shape
        x_e = einops.rearrange(x, "b d h w -> (b h w) d")

        distances = ((self.embedding.weight[:, None, :] - x_e[None, :, :]) ** 2).sum(dim = -1)
        latent_indices =  torch.argmin(distances, dim = 0)

        x_q = self.embedding.weight[latent_indices] #representation of x in the embedding space
        x_q = einops.rearrange(x_q, "(b h w) d -> b d h w", h = height, w = weight)

        loss = F.mse_loss(x_q, x.detach()) + self.beta * F.mse_loss(x_q.detach(), x)

        # estimator trick
        x_q = x + (x_q - x).detach()

        return x_q, loss, latent_indices          

        
class VQVAE(nn.Module):
    def __init__(self, config):
        super(VQVAE, self).__init__()

        self.batch_size = config["batch_size"]
        self.n_channels = config["n_channels"]
        self.channels = config["channels"]
        self.latent_dim = config["latent_dim"]
        self.n_embedding = config["n_embedding"]

        self.r = len(self.channels) #reduction factor (nb of conv blocks)

        if self.r > 2:
            self.reshape = True
            self.reshape_factor = 2**(self.r - 2)
        else:
            self.reshape = False            

        self.encoder = EncoderBlock(self.n_channels, self.channels + [self.latent_dim])
        self.vq = VectorQuantizer(self.n_embedding, self.latent_dim)
        
        self.decoder = DecoderBlock(self.n_channels, self.channels + [self.latent_dim])
        
        self.final_conv = nn.Conv2d(self.n_channels, self.n_channels,  kernel_size = 1, padding = "same")

    def forward(self, x):
        
        if self.reshape:
            x = F.interpolate(x, scale_factor = self.reshape_factor, mode='bilinear')

        x = self.encoder(x)
        x, vq_loss, latent_indices = self.vq(x)
        x = self.decoder(x)

        x = self.final_conv(x)
        if self.reshape:
            x = F.interpolate(x, scale_factor = 1/self.reshape_factor, mode='bilinear')

        return x, vq_loss, latent_indices

    def generate(self, p_dist, size=(28,28)):

        if self.r > 2:
            size = (size[0] * self.reshape_factor, size[1] * self.reshape_factor)            

        height, width = size
        latent_height = height // (2**self.r)
        latent_width = width // (2**self.r)

        latent_indices = torch.multinomial(p_dist, latent_height* latent_width, replacement=True)   

        embeddings = self.vq.embedding.weight[latent_indices]

        embeddings = einops.rearrange(embeddings, "(b h w) d -> b d h w", h=latent_height, w=latent_width)
       
        generated_image = self.decoder(embeddings)
        generated_image = self.final_conv(generated_image)

        if self.reshape:
            generated_image = F.interpolate(generated_image, scale_factor = 1/self.reshape_factor, mode='bilinear')
        
        return generated_image[0][0]

        