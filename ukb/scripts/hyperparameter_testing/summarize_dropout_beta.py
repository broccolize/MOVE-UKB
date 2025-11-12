import pandas as pd
df = pd.read_csv("results/tune_model/reconstruction_stats.tsv", sep="\t")
sub = df[(df["split"]=="test") &
         (df["task.model.num_latent"]==32) &
         (df["task.model.num_hidden"]=="[1024,512]")]
rank = (sub.groupby(["task.model.dropout","task.model.beta"])
          ["med"].mean().sort_values(ascending=False).reset_index())
print("\n=== Regularization sweep results (test split) ===")
print(rank.to_string(index=False))