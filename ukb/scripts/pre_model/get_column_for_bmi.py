import pandas as pd

cont = pd.read_csv("data/ukb_continuous.tsv", sep="\t")
print("CONT:", [c for c in cont.columns if "bmi" in c.lower() or "210" in c])

batch = pd.read_csv("data/ukb_batch.tsv", sep="\t")
print("BATCH:", [c for c in batch.columns if "bmi" in c.lower() or "210" in c])

prot = pd.read_csv("data/ukb_proteomics.tsv", sep="\t", nrows=1)
print("PROT:", prot.columns.tolist()[:20])
