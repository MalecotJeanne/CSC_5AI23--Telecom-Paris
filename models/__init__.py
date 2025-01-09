from models.vqvae import VQVAE

models = {
    "vqvae": VQVAE,
}

def init_model(name, *args, **kwargs):
    if name not in models.keys():
        raise KeyError("Unknown models: {}".format(name)) 
    return models[name](*args, **kwargs)
