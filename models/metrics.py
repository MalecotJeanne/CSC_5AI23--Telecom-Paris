import torch
import torch.nn as nn
import torch.nn.functional as F



def vae_loss(reconstructed_x, x, mu, logvar):
    
    BCE = nn.functional.binary_cross_entropy_with_logits(reconstructed_x, x, reduction='sum')
    KL_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KL_divergence

def vqvae_loss(reconstructed_x, x, vq_loss):

    reconstruction_loss = 0.5 * F.mse_loss(reconstructed_x, x) + 0.5 * (1 - SSIM(reconstructed_x, x))
    
    return reconstruction_loss + vq_loss


def SSIM(reconstructed_x, x, k1=0.01, k2=0.03, L=255):

    mu_x = torch.mean(x)
    mu_y = torch.mean(reconstructed_x)

    sigma_x = torch.std(x)
    sigma_y = torch.std(reconstructed_x)
    sigma_xy = torch.mean((x - mu_x) * (reconstructed_x - mu_y))

    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    
    luminance = (2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)
    contrast = (2 * sigma_x * sigma_y + C2) / (sigma_x ** 2 + sigma_y ** 2 + C2)
    structure = (sigma_xy + C3) / (sigma_x * sigma_y + C3)

    return luminance * contrast * structure
