import pandas as pd

metrics = pd.read_csv("results/latent_space/reconstruction_metrics.tsv", sep="\t")
print(metrics)
