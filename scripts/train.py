import torch
from tqdm.auto import tqdm
import torch.optim as optim 
from torch.utils.data import random_split
import copy

from IPython.display import clear_output, display
from ipywidgets import Output

from models.metrics import vqvae_loss
from models import init_model


def train_model(model_name, train_set, config, device="cpu"):

    train_data, val_data = random_split(train_set, [0.8, 0.2])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config['model']['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config['model']['batch_size'], shuffle=False)

    model = init_model(model_name, config['model']).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    model.to(device)
    
    epoch_output = Output()
    display(epoch_output)

    for epoch in range(config['n_epochs']):
        model.train()
        running_loss = 0.0

        with epoch_output:
            for i, data in enumerate(tqdm(train_loader, 
                        desc=f"Epoch {epoch+1}/{config['n_epochs']} - Training")):
                
                x, _ = data
                x = x.to(device)

                optimizer.zero_grad()
                
                reconstructed_x, vq_loss,_ = model(x)
                
                loss = vqvae_loss(reconstructed_x, x, vq_loss)
                loss.backward()
                optimizer.step()
                
                
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)

            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader, desc='Validation')):
                    x, _ = data
                    x = x.to(device)
                    
                    reconstructed_x, vq_loss,_ = model(x)

                    loss = vqvae_loss(reconstructed_x, x, vq_loss)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
        
            clear_output()

    model_dict = {"model": copy.deepcopy(model), "train_loss": avg_loss, "val_loss": avg_val_loss, "config": copy.deepcopy(config)}
    
    del model

    return model_dict
