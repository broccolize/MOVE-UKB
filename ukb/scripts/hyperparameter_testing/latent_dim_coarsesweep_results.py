import pandas as pd

df = pd.read_csv("results/tune_model/reconstruction_stats.tsv", sep="\t")

# focus only on test split
test = df[df["split"]=="test"].copy()

# average the median value across all datasets/metrics for each job
group_cols = [
    "task.model.num_latent", 
    "task.model.num_hidden", 
    "task.model.beta" if "task.model.beta" in df.columns else None
]
group_cols = [c for c in group_cols if c in df.columns]

summary = (test.groupby(group_cols)["med"].mean().sort_values(ascending=False))
print(summary.to_string())
print("\nTop configuration:")
print(summary.head(1))
