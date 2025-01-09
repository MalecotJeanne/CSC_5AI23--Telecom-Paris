import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.decomposition import PCA

import torch.nn.functional as F

def reconstruct(model, test_set, device='cuda'):

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    
    n_embeddings = model.n_embedding
    label_distribution = torch.zeros((10, n_embeddings))
    model.eval()

    x_dicts = []
    latent_vectors = []
    latent_labels = []

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

            latent_vector = latent_indices.cpu().numpy().flatten()
            latent_vectors.append(latent_vector)
            latent_labels.append(label)
            
            x_dicts.append({"original": x, "reconstructed": reconstructed_x, "label": label, "latent_indices": latent_indices.cpu().numpy()})

    latent_vectors = np.array(latent_vectors)
    print("latent_vectors", latent_vectors.shape)

    reduced_indices = PCA(n_components=2).fit_transform(latent_vectors)
    print("reduced_indices", reduced_indices.shape)

    return x_dicts, label_distribution, reduced_indices, latent_labels


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


def plot_latent_space(reduced_embeddings, latent_labels):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=latent_labels,
        cmap='tab10',
        s=10,
        alpha=0.8
    )
    plt.colorbar(scatter, label="Labels")
    plt.title("Latent Space Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()
