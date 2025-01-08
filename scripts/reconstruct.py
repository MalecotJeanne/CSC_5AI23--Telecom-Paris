import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch.nn.functional as F


def reconstruct(model, test_set, device='cuda'):

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    
    n_embeddings = model.n_embedding

    label_distribution = torch.zeros((10, n_embeddings))
    model.eval()
    x_dicts = []
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Reconstructing images", position=0, leave=True):
            x, label = data
            x = x.to(device)
            
            reconstructed_x, _, latent_indices = model(x)
            
            reconstructed_x = reconstructed_x.cpu()
            x = x.cpu()

            label = label.item()
            

            for indice in (latent_indices):
                label_distribution[label, indice] += 1
            
            x_dicts.append({"original": x, "reconstructed": reconstructed_x, "label": label, "latent_indices": latent_indices.cpu().numpy()})


    # label_distribution = F.softmax(label_distribution/10000, dim=1)
    # label_distribution = label_distribution / label_distribution.sum(dim=1, keepdim=True)

    return x_dicts, label_distribution

def show_recon(x_dicts):
    random_indices = random.sample(range(len(x_dicts)), 4)

    fig, axes = plt.subplots(2, 4, figsize=(14, 8)) 
    for i, idx in enumerate(random_indices):
        original_image = x_dicts[idx]['original']
        reconstructed_image = x_dicts[idx]['reconstructed']

        row = i // 2
        col = (i % 2) * 2  

        axes[row, col].imshow(original_image.squeeze(), cmap='gray')
        axes[row, col].set_title(f"Original")
        axes[row, col].axis('off')
        
        axes[row, col + 1].imshow(reconstructed_image.squeeze(), cmap='gray')
        axes[row, col + 1].set_title(f"Reconstructed")
        axes[row, col + 1].axis('off')

    plt.tight_layout()
    plt.show()