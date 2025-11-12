import pandas as pd

# Load all reconstruction stats
df = pd.read_csv("results/tune_model/reconstruction_stats.tsv", sep="\t")

# Filter for test split and your tuned hyperparams
sub = df[
    (df["split"] == "test") &
    (df["task.model.num_hidden"] == "[1024,512]") &
    (df["task.model.dropout"] == 0.2) &
    (df["task.model.beta"] == 0.0005)
]

# Group by latent and other key params
rank = (
    sub.groupby(["task.model.num_latent", "task.batch_size", "task.training_loop.num_epochs"])["med"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

print("\n=== Reconstruction accuracy across configurations (higher = better) ===")
print(rank.to_string(index=False))