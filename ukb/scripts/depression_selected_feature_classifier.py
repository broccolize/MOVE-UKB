import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
TOP_LIST = [20, 50, 100, 200, 500,1000 , 2000]   # <- the sweep values
PROTEOMICS_FILE = "ukb/data/ukb_proteomics.tsv"
ICD_FILE = "ukb/UKB_ICD10F_diagnosis2.csv"
SIG_FILE = "significant_depression_proteins.tsv"


def load_data():
    print("Loading proteomics...")
    proteins = pd.read_csv(PROTEOMICS_FILE, sep="\t")
    proteins = proteins.rename(columns={"X.SampleID": "eid"})
    
    print("Loading ICD...")
    icd = pd.read_csv(ICD_FILE)

    # Create binary depression flag
    icd["depression"] = (
        (icd["F32_diagnosis"] == True) |
        (icd["F33_diagnosis"] == True)
    ).astype(int)

    print("Loading significant proteins...")
    sig = pd.read_csv(SIG_FILE, sep="\t")

    return proteins, icd, sig


def preprocess(proteins, icd, sig, topN):
    """Return X, y for a given number of top proteins"""
    # Metadata columns—exclude from protein list
    metadata_cols = [
        "eid", "batch", "fastingtime_0_0",
        "bloodsample_date_0_0", "PlateID_0_0", "WellID_0_0"
    ]
    real_protein_cols = [c for c in proteins.columns if c not in metadata_cols]

    # Sort and choose Top N proteins by effect size
    selected_topN = (
        sig.sort_values("effect_size", ascending=False)
           .head(topN)["protein"].tolist()
    )

    # Keep only proteins that actually exist in the data
    selected_proteins = [p for p in selected_topN if p in real_protein_cols]

    # Merge labels onto proteomics
    merged = proteins.merge(icd[["eid", "depression"]], on="eid", how="inner")

    X = merged[selected_proteins]
    y = merged["depression"]

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Scale
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y, selected_proteins


def evaluate(X, y):
    """Train + evaluate classifier"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)

    prob = clf.predict_proba(X_test)[:, 1]
    pred = clf.predict(X_test)

    auc = roc_auc_score(y_test, prob)
    acc = accuracy_score(y_test, pred)

    return auc, acc


# -------------------------------------------------------------
# MAIN SWEEP
# -------------------------------------------------------------
proteins, icd, sig = load_data()

results = []

for topN in TOP_LIST:
    print("\n=======================================")
    print(f"Running sweep for Top {topN} proteins")
    print("=======================================")

    X, y, selected_proteins = preprocess(proteins, icd, sig, topN)
    auc, acc = evaluate(X, y)

    print(f"Top {topN}: AUC={auc:.4f}, Accuracy={acc:.4f}")

    results.append((topN, auc, acc))


# -------------------------------------------------------------
# OUTPUT SUMMARY
# -------------------------------------------------------------
print("\n\n====================== FINAL RESULTS ======================")
print("TopN\tAUC\tAccuracy")
for topN, auc, acc in results:
    print(f"{topN}\t{auc:.4f}\t{acc:.4f}")

print("============================================================")