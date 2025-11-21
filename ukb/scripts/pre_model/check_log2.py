import pandas as pd
import numpy as np

df = pd.read_csv("data/ukb_proteomics.tsv", sep="\t")
X = df.drop(columns=[df.columns[0]])

summary = {
    "min_all": X.min().min(),
    "max_all": X.max().max(),
    "mean_all": X.values.mean(),
    "median_all": float(np.median(X.values)),
    "pct_zeros": float((X==0).sum().sum() / X.size * 100),
    "pct_neg": float((X<0).sum().sum() / X.size * 100)
}
print(summary)

print(X.sample(n=5, axis=1, random_state=0).describe().T.iloc[:, :6])

