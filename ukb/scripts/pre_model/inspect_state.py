import torch

state = torch.load("results/latent_space/model.pt", map_location="cpu")

for k,v in state.items():
    if hasattr(v, "shape"):
        print(k, v.shape)
