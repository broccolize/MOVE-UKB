import pandas as pd
df = pd.read_csv("results/tune_model/reconstruction_stats.tsv", sep="\t")
sub = df[(df["split"]=="test") &
         (df["task.model.num_latent"].isin([32,64])) &
         (df["task.model.num_hidden"].isin(["[256,128]","[512,256]","[1024,512]"]))]

rank = (sub.groupby(["task.model.num_latent","task.model.num_hidden"])
          ["med"].mean().sort_values(ascending=False).reset_index())
print("\n=== Hidden-width × latent results (test split) ===")
print(rank.to_string(index=False))