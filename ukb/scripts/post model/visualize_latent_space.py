import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

# === Load latent matrix ===
latent = pd.read_csv("results/latent_space/latent_space_full.tsv", sep="\t", index_col=0)
print("Latent matrix:", latent.shape)

# === Load optional metadata (e.g., batch) ===
batch = np.load("interim_data/ukb_batch.npy")
batch_labels = batch.argmax(axis=-1).squeeze()
meta = pd.DataFrame({"batch": batch_labels}, index=latent.index)

# === Reduce to 2D ===
method = "UMAP"   # choose "PCA", "TSNE", or "UMAP"

if method == "PCA":
    reducer = PCA(n_components=2)
    coords = reducer.fit_transform(latent)
elif method == "TSNE":
    reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    coords = reducer.fit_transform(latent)
elif method == "UMAP":
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="euclidean", random_state=42)
    coords = reducer.fit_transform(latent)

# === Plot ===
plt.figure(figsize=(8,6))
sns.scatterplot(x=coords[:,0], y=coords[:,1], hue=meta["batch"], 
                palette="tab10", s=10, alpha=0.7, edgecolor=None)
plt.title(f"{method} projection of latent space (colored by batch)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(title="Batch", bbox_to_anchor=(1.05,1), loc="upper left")
plt.tight_layout()
plt.savefig("results/latent_space_umap.png", dpi=300)
plt.close()
plt.show()