import pandas as pd
lat = pd.read_csv("results/latent_space/latent_space.tsv", sep="\t")
print(lat.shape)
print(lat.head())
