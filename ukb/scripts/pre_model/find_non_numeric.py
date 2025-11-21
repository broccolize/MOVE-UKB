import pandas as pd

df = pd.read_csv("data/ukb_proteomics.tsv", sep="\t")
X = df.drop(columns=["X.SampleID"])
bad = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
print("non numeric columns:", bad)
