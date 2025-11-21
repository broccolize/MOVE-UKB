import pandas as pd

l = pd.read_csv("results/latent_space/latent_space_full.tsv", sep="\t")
print(l.columns[:10])

i = pd.read_csv("UKB_ICD10F_diagnosis2.csv")
print(i.columns[:10])