import random
import matplotlib.pyplot as plt

def show_reconstructions(x_dicts):
    random_indices = random.sample(range(len(x_dicts)), 4)

    fig, axes = plt.subplots(2, 4, figsize=(14, 8)) 
    for i, idx in enumerate(random_indices):
        original_image = x_dicts[idx]['original']
        reconstructed_image = x_dicts[idx]['reconstructed']

        row = i // 2
        col = (i % 2) * 2  

        axes[row, col].imshow(original_image.squeeze(), cmap='gray')
        axes[row, col].set_title(f"Original {idx}")
        axes[row, col].axis('off')
        
        axes[row, col + 1].imshow(reconstructed_image.squeeze(), cmap='gray')
        axes[row, col + 1].set_title(f"Reconstructed {idx}")
        axes[row, col + 1].axis('off')

    plt.tight_layout()
    plt.show()