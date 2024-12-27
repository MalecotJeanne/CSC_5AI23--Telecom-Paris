#!/usr/bin/env python
# coding: utf-8

# # VQVAE for Image Generation - FashionMNIST Dataset
# **Author:** Jeanne Malécot

# ## Getting Started

# In[1]:


# useful imports
import os
import copy

import torch
import torchvision
import torchvision.transforms as transforms
import torchinfo
import random

import matplotlib.pyplot as plt

from scripts.train import train_model
from scripts.reconstruct import reconstruct, show_recon
from models.vqvae import VQVAE


## results directory
results_dir = "Results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# ##### Load FashionMNIST dataset

# load FashionMNIST
train_set = torchvision.datasets.FashionMNIST(
    root="data", train=True, download=True, transform=transforms.ToTensor()
)
test_set = torchvision.datasets.FashionMNIST(
    root="data", train=False, download=True, transform=transforms.ToTensor()
)

### Training VQVAE for reconstruction

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# config
basic_config = {
    "n_epochs": 30,
    "lr": 0.001,
    "model": {
        "batch_size": 100,
        "n_channels": 1,
        "channels": [64, 128],
        "latent_dim": 20,
        "n_embedding": 20,
    },
}

# Model architecture:

vqvae = vqvae = VQVAE(basic_config["model"]).to(device)
print(torchinfo.summary(vqvae, (1, 1, 28, 28), device=str(device)))

# #### Fine tuning of the VQ-VAE
# grid search

model_dicts = []
lr_list = [0.03, 0.01, 0.001]
batch_size_list = [50, 100, 200]
channels = [[16, 32], [32, 64], [64, 128], [128, 256]]
latent_dim = [k for k in range(10, 101, 10)]
n_embedding_list = [k for k in range(10, 501, 10)]

grid = [
    (lr, batch_size, channel, latent, n_embedding)
    for lr in lr_list
    for batch_size in batch_size_list
    for channel in channels
    for latent in latent_dim
    for n_embedding in n_embedding_list
]

for i, (lr, batch_size, channel, latent, n_embedding) in enumerate(grid):
    config = copy.deepcopy(basic_config)
    config["lr"] = lr
    config["model"]["batch_size"] = batch_size
    config["model"]["channels"] = channel
    config["model"]["latent_dim"] = latent
    config["model"]["n_embedding"] = n_embedding
    model_dicts.append(train_model("vqvae", train_set, config, device))

# for i, n_embeddings in enumerate(n_embedding_list):
#     config = copy.deepcopy(basic_config)
#     config['model']['n_embedding'] = n_embeddings
#     model_dicts.append(train_model("vqvae", train_set, config, device))


# select model with lower accuracy
sorted_dicts = sorted(model_dicts, key=lambda x: x["val_loss"])

print("\n\n############################################\n\n")
print("Best 5 models:\n")
print(sorted_dicts[:5])
print("\n\n############################################\n\n")

best_config = sorted_dicts[0]["config"]
best_model = sorted_dicts[0]["model"]

# save 5 firsts models
for i in range(5):
    torch.save(sorted_dicts[i]["model"].state_dict(), f"{results_dir}/model_{i}.pth")
# save 5 firsts configs
for i in range(5):
    with open(f"{results_dir}/config_{i}.txt", "w") as f:
        f.write(str(sorted_dicts[i]["config"]))
