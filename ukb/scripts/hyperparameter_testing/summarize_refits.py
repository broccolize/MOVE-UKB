import pandas as pd
df = pd.read_csv("results/tune_model/stability_stats.tsv", sep="\t")
# keep only the metric we care about
sub = df[df["metric"]=="mean_diff_cosine_similarity"]
print(sub[["task.model.num_latent","task.model.num_hidden","task.model.beta","task.model.dropout","med","mean"]])