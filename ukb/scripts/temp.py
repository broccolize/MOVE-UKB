import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. Load raw proteomics file
# ============================================================
proteins = pd.read_csv("UKB_Protein_BL_withCov.csv")

# Inspect columns
print("Proteomics columns:", proteins.columns[:10])
print("Shape:", proteins.shape)

# Assume participant ID column is named "eid" or "EID"
# Adjust if needed:
if "eid" in proteins.columns:
    proteins.rename(columns={"eid": "EID"}, inplace=True)

# ============================================================
# 2. Load ICD depression labels (F32/F33)
# ============================================================
icd = pd.read_csv("UKB_ICD10F_diagnosis2.csv")

# Normalize column names
icd.rename(columns={"eid": "EID"}, inplace=True)

# Create depression label
icd["depression"] = (
    icd["F32_diagnosis"].astype(bool) |
    icd["F33_diagnosis"].astype(bool)
).astype(int)

print("Depression prevalence:", icd["depression"].mean())

# ============================================================
# 3. Merge proteomics + depression labels
# ============================================================
merged = proteins.merge(icd[["EID", "depression"]], on="EID", how="inner")
print("Merged shape:", merged.shape)

# ============================================================
# 4. Prepare protein matrix
# ============================================================
# Identify protein columns (exclude ID and covariates)
exclude = {"EID", "PlateID_0_0", "batch"}
protein_cols = [c for c in merged.columns if c not in exclude and c != "depression"]

print("Number of protein columns:", len(protein_cols))

# Split case/control
cases = merged[merged["depression"] == 1]
controls = merged[merged["depression"] == 0]

# ============================================================
# 5. Run t-tests for each protein
# ============================================================
results = []

for p in protein_cols:
    case_vals = cases[p].replace([np.inf, -np.inf], np.nan).dropna()
    ctrl_vals = controls[p].replace([np.inf, -np.inf], np.nan).dropna()

    # Skip proteins with little variation
    if len(case_vals) < 20 or len(ctrl_vals) < 20:
        continue

    tstat, pval = ttest_ind(case_vals, ctrl_vals, equal_var=False)

    results.append({
        "protein": p,
        "mean_case": case_vals.mean(),
        "mean_control": ctrl_vals.mean(),
        "effect_size": case_vals.mean() - ctrl_vals.mean(),
        "p_value": pval
    })

df = pd.DataFrame(results)

# ============================================================
# 6. FDR (False Discovery Rate) correction
# ============================================================
df["p_adj"] = multipletests(df["p_value"], method="fdr_bh")[1]

# Sort by p-value
df = df.sort_values("p_value")

# Save full results
df.to_csv("protein_depression_associations.tsv", sep="\t", index=False)

print("\nTop hits:")
print(df.head())

# ============================================================
# 7. Extract significant proteins (p < 0.05)
# ============================================================
sig = df[df["p_adj"] < 0.05]
sig.to_csv("significant_depression_proteins.tsv", sep="\t", index=False)

print("\nNumber of significant proteins:", len(sig))

# ============================================================
# 8. Volcano plot
# ============================================================
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df,
    x="effect_size",
    y=-np.log10(df["p_value"]),
    hue=df["p_adj"] < 0.05,
    palette={True: "red", False: "gray"},
    alpha=0.7
)

plt.title("Volcano Plot: Depression-associated Proteins")
plt.xlabel("Effect Size (Case − Control)")
plt.ylabel("-log10(p)")
plt.legend(["Not significant", "Significant (FDR < 0.05)"])
plt.tight_layout()
plt.savefig("volcano_plot_depression.png", dpi=300)
plt.close()

print("\nDone! Files saved:")
print(" - protein_depression_associations.tsv")
print(" - significant_depression_proteins.tsv")
print(" - volcano_plot_depression.png")