import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Local Benjamini–Hochberg FDR (no statsmodels needed)
# ---------------------------------------------------------
def fdr_bh(pvals):
    """
    Benjamini-Hochberg False Discovery Rate correction.
    pvals: array-like of p-values
    Returns: array of FDR-adjusted p-values.
    """
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.arange(1, n + 1)
    adj = np.empty(n, dtype=float)

    # Basic BH formula
    adj[order] = pvals[order] * n / ranks

    # Enforce monotone decreasing when going from largest to smallest p
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    return adj


# ---------------------------------------------------------
# 1. Load raw proteomics and ICD diagnosis
# ---------------------------------------------------------
prot_path = "UKB_Protein_BL_withCov.csv"
icd_path = "UKB_ICD10F_diagnosis2.csv"

proteins = pd.read_csv(prot_path)
icd = pd.read_csv(icd_path)

print("Proteomics shape:", proteins.shape)
print("Proteomics columns (first 10):", list(proteins.columns[:10]))
print("ICD shape:", icd.shape)
print("ICD columns (first 10):", list(icd.columns[:10]))

# Normalise ID column names
if "eid" in proteins.columns:
    proteins.rename(columns={"eid": "EID"}, inplace=True)
if "X.SampleID" in proteins.columns:
    proteins.rename(columns={"X.SampleID": "EID"}, inplace=True)

if "eid" in icd.columns:
    icd.rename(columns={"eid": "EID"}, inplace=True)

# ---------------------------------------------------------
# 2. Build depression label from F32 / F33
# ---------------------------------------------------------
if "F32_diagnosis" not in icd.columns or "F33_diagnosis" not in icd.columns:
    raise ValueError("ICD file must contain 'F32_diagnosis' and 'F33_diagnosis' columns.")

icd["depression"] = (
    icd["F32_diagnosis"].astype(bool) | icd["F33_diagnosis"].astype(bool)
).astype(int)

print("Depression prevalence (ICD file):", icd["depression"].mean())

# Keep only necessary columns from ICD
icd_keep = icd[["EID", "depression"]]

# ---------------------------------------------------------
# 3. Merge proteomics + depression
# ---------------------------------------------------------
merged = proteins.merge(icd_keep, on="EID", how="inner")
print("Merged shape:", merged.shape)

# ---------------------------------------------------------
# 4. Identify protein columns (exclude ID, covariates, label)
# ---------------------------------------------------------
exclude_cols = {"EID", "batch", "PlateID_0_0", "depression"}

candidate_cols = [c for c in merged.columns if c not in exclude_cols]

# Keep only numeric columns as proteins
protein_cols = [
    c for c in candidate_cols
    if np.issubdtype(merged[c].dtype, np.number)
]

print("Number of numeric candidate protein columns:", len(protein_cols))

# ---------------------------------------------------------
# 5. Split cases and controls
# ---------------------------------------------------------
cases = merged[merged["depression"] == 1]
controls = merged[merged["depression"] == 0]

print("Cases:", len(cases), "Controls:", len(controls))

# ---------------------------------------------------------
# 6. Run t-tests protein-by-protein
# ---------------------------------------------------------
results = []

for p in protein_cols:
    case_vals = cases[p].replace([np.inf, -np.inf], np.nan).dropna()
    ctrl_vals = controls[p].replace([np.inf, -np.inf], np.nan).dropna()

    # Skip if too few observations in either group
    if len(case_vals) < 20 or len(ctrl_vals) < 20:
        continue

    tstat, pval = ttest_ind(case_vals, ctrl_vals, equal_var=False)

    results.append({
        "protein": p,
        "n_case": len(case_vals),
        "n_control": len(ctrl_vals),
        "mean_case": case_vals.mean(),
        "mean_control": ctrl_vals.mean(),
        "effect_size": case_vals.mean() - ctrl_vals.mean(),
        "p_value": pval,
    })

df = pd.DataFrame(results)
print("Tested proteins:", len(df))

if df.empty:
    raise RuntimeError("No proteins passed basic filtering (check column names and data).")

# ---------------------------------------------------------
# 7. FDR correction
# ---------------------------------------------------------
df["p_adj"] = fdr_bh(df["p_value"])

# Sort by raw p-value
df = df.sort_values("p_value")

# Save full results
df.to_csv("protein_depression_associations.tsv", sep="\t", index=False)

print("\nTop 10 proteins by raw p-value:")
print(df.head(10))

# ---------------------------------------------------------
# 8. Extract significant hits (FDR < 0.05)
# ---------------------------------------------------------
sig = df[df["p_adj"] < 0.05].copy()
sig = sig.sort_values("p_adj")

sig.to_csv("significant_depression_proteins.tsv", sep="\t", index=False)

print("\nNumber of proteins with FDR < 0.05:", len(sig))
if len(sig) > 0:
    print("\nTop 10 significant proteins (FDR < 0.05):")
    print(sig.head(10))

# ---------------------------------------------------------
# 9. Optional: Volcano plot
# ---------------------------------------------------------
try:
    plt.figure(figsize=(10, 7))
    x = df["effect_size"]
    y = -np.log10(df["p_value"].clip(lower=1e-300))  # avoid log(0)
    sig_mask = df["p_adj"] < 0.05

    plt.scatter(x[~sig_mask], y[~sig_mask], c="grey", alpha=0.5, s=8, label="Not significant")
    plt.scatter(x[sig_mask], y[sig_mask], c="red", alpha=0.7, s=10, label="FDR < 0.05")

    plt.axhline(-np.log10(0.05), color="blue", linestyle="--", linewidth=1, label="p = 0.05")
    plt.xlabel("Effect size (mean_case − mean_control)")
    plt.ylabel("-log10(p-value)")
    plt.title("Volcano plot: depression-associated proteins")
    plt.legend()
    plt.tight_layout()
    plt.savefig("volcano_plot_depression.png", dpi=300)
    plt.close()
    print("\nSaved volcano plot to 'volcano_plot_depression.png'")
except Exception as e:
    print("Could not generate volcano plot:", e)