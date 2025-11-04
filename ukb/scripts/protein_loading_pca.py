import torch
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

state = torch.load("results/latent_space/model.pt", map_location="cpu")

W = state["out.weight"].detach().cpu().numpy()   # shape (3925, 256) in saved model

n_batch = 1
n_date  = 1
n_prot  = 2922

prot_start = n_batch + n_date              # =2
prot_end   = prot_start + n_prot           # =2924

W_prot = W[prot_start:prot_end, :]         # (2922, 256)

p = PCA(n_components=2)
pc = p.fit_transform(W_prot)

plt.figure(figsize=(8,6))
plt.scatter(pc[:,0], pc[:,1], s=3)
plt.title("Protein clustering in latent loadings")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
