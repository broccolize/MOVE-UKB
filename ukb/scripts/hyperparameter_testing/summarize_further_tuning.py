import pandas as pd

# Load the new stability results
df = pd.read_csv("results/tune_model/stability_stats.tsv", sep="\t")

# Filter only the stability metric we care about
sub = df[df["metric"] == "mean_diff_cosine_similarity"].copy()

# Group by the hyperparams you varied (adjust if you added others)
rank = (
    sub.groupby([
        "task.model.num_latent",
        "task.model.num_hidden",
        "task.model.dropout",
        "task.model.beta"
    ])["med"]
    .mean()
    .sort_values(ascending=True)
    .reset_index()
)

print("\n=== Stability tuning results (lower = better) ===")
print(rank.to_string(index=False))