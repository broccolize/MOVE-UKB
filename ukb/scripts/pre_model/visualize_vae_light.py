"""
Lightweight VAE visualization for MOVE-UKB.
Draws only latent → decoder_hidden → output layers with sampled edges.
Finishes in seconds even for large models.
"""

import torch
import matplotlib.pyplot as plt
import networkx as nx
import random
from pathlib import Path

# ---- configuration ----
model_path = Path("ukb/results/latent_space/model_16_0.pt")  # rename if needed
output_path = Path("ukb/results/latent_space/vae_light.png")
sample_edges = 2000  # number of edges to display
node_size = 80
figsize = (15, 10)
# ------------------------

# load weights
weights = torch.load(model_path, map_location="cpu")

# create graph
G = nx.Graph()

# extract relevant layers
decoder_w = weights["decoderlayers.0.weight"].cpu().numpy()
out_w = weights["out.weight"].cpu().numpy()

n_latent, n_hidden = decoder_w.shape[1], decoder_w.shape[0]
n_output = out_w.shape[0]

# add nodes
for i in range(n_latent):
    G.add_node(f"latent_{i}", pos=(0, i), color=0.2)
for j in range(n_hidden):
    G.add_node(f"hidden_{j}", pos=(1, j / 4), color=0.4)
for k in range(n_output):
    if k % 40 == 0:  # sample output nodes for visibility
        G.add_node(f"out_{k}", pos=(2, k / 80), color=0.6)

# add sampled edges from latent→hidden
latent_edges = [
    (f"latent_{i}", f"hidden_{j}", decoder_w[j, i])
    for i in range(n_latent)
    for j in range(n_hidden)
]
latent_edges = random.sample(latent_edges, min(sample_edges, len(latent_edges)))

# add sampled edges from hidden→output
output_edges = [
    (f"hidden_{j}", f"out_{k}", out_w[k, j])
    for j in range(n_hidden)
    for k in range(0, n_output, 40)
]
output_edges = random.sample(output_edges, min(sample_edges, len(output_edges)))

# add edges to graph
for (a, b, w) in latent_edges + output_edges:
    G.add_edge(a, b, weight=w)

# plot
plt.figure(figsize=figsize)
pos = nx.get_node_attributes(G, "pos")
edge_colors = [w for (_, _, w) in G.edges(data="weight")]
node_colors = [nx.get_node_attributes(G, "color")[n] for n in G.nodes()]
nx.draw(
    G,
    pos=pos,
    node_size=node_size,
    node_color=node_colors,
    edge_color=edge_colors,
    edge_cmap=plt.cm.seismic,
    width=0.5,
)
plt.title("MOVE-UKB lightweight VAE connectivity (latent→decoder→output)")
plt.tight_layout()
plt.savefig(output_path, dpi=150)
print(f"✅ Saved lightweight visualization to {output_path}")
