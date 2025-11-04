import pandas as pd
import numpy as np
from pathlib import Path

IN = Path("UKB_Protein_BL_withCov.csv")
OUT = Path("ukb/data")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN)

id_col = "eid"

# -------------------------
# choose covariates YOU want to treat as continuous
# these 3 seem numeric / timestamps
continuous_covariates = [
    "fastingtime_0_0",
    "ProteinNum_0_0"
]

# choose covariates to treat as categorical
categorical_covariates = [
    "batch",
    "bloodsample_date_0_0"
]


# -------------------------
# identify proteomics cols = numeric but not in covariates and not ID
all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
proteomics_cols = [c for c in all_numeric
                   if c not in continuous_covariates and c != id_col]

# -------------------------
# sample IDs
ids = df[[id_col]].drop_duplicates().sort_values(id_col)
ids.to_csv(OUT/"ukb_ids.txt", index=False, header=False)

# -------------------------
# proteomics
prot = df[[id_col] + proteomics_cols].copy()
prot.insert(0, "X.SampleID", prot.pop(id_col))

# mask crazy sentinel values
prot.iloc[:, 1:] = prot.iloc[:, 1:].mask(prot.iloc[:, 1:] > 1e6)

prot.to_csv(OUT/"ukb_proteomics.tsv", sep="\t", index=False)

# -------------------------
# continuous covariates
cont = df[[id_col] + continuous_covariates].copy()
cont.insert(0, "X.SampleID", cont.pop(id_col))
cont.to_csv(OUT/"ukb_continuous.tsv", sep="\t", index=False)

# -------------------------
# categorical covariates
for c in categorical_covariates:
    t = df[[id_col, c]].copy()
    t.insert(0, "X.SampleID", t.pop(id_col))
    t.to_csv(OUT/f"ukb_{c}.tsv", sep="\t", index=False)

print("DONE — wrote:")
print(" → ukb_ids.txt")
print(" → ukb_proteomics.tsv")
print(" → ukb_continuous.tsv")
print(" → [categorical].tsv")
