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

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['model']['batch_size'], shuffle=True)

    model = init_model(model_name, config['model']).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    epoch_output = Output()
    display(epoch_output)

    
    losses_dict = {"loss": [], "reconstruction_loss": [], "vq_loss": []}

    for epoch in range(config['n_epochs']):
        model.train()
        running_loss = 0.0
        running_reconstruction_loss = 0.0
        running_vq_loss = 0.0

        with epoch_output:
            for i, data in enumerate(tqdm(train_loader, 
                        desc=f"Epoch {epoch+1}/{config['n_epochs']} - Training")):
                
                x, _ = data
                x = x.to(device)

                optimizer.zero_grad()
                
                reconstructed_x, vq_loss,_ = model(x)
                
                loss, reconstruction_loss, vq_loss = vqvae_loss(reconstructed_x, x, vq_loss, config['alpha'])
                loss.backward()
                optimizer.step()     

                running_loss += loss.item()
                running_reconstruction_loss += reconstruction_loss.item()
                running_vq_loss += vq_loss.item()


            avg_loss = running_loss / len(train_loader)
            avg_reconstruction_loss = running_reconstruction_loss / len(train_loader)
            avg_vq_loss = running_vq_loss / len(train_loader)
            model.eval()

            losses_dict["loss"].append(avg_loss)
            losses_dict["reconstruction_loss"].append(avg_reconstruction_loss)
            losses_dict["vq_loss"].append(avg_vq_loss)
            
            scheduler.step()
        
            clear_output()

    model_dict = {"model": copy.deepcopy(model), "train_loss": avg_loss, "config": copy.deepcopy(config)}
    
    del model

    return model_dict, losses_dict
