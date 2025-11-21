import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# 1. Load raw proteomics
proteins = pd.read_csv("data/ukb_proteomics.tsv", sep="\t")
proteins = proteins.rename(columns={"X.SampleID": "eid"})

# 2. Load ICD labels
icd = pd.read_csv("UKB_ICD10F_diagnosis2.csv")
icd.columns = [c.strip().upper() for c in icd.columns]

icd["depression"] = (
    icd["F32_DIAGNOSIS"].astype(str).str.upper().eq("TRUE") |
    icd["F33_DIAGNOSIS"].astype(str).str.upper().eq("TRUE")
).astype(int)

# 3. Merge
merged = proteins.merge(icd[["EID", "depression"]],
                        left_on="eid",
                        right_on="EID",
                        how="inner")

print("Merged shape:", merged.shape)

# 4. Feature matrix and target
X = merged.drop(columns=["eid", "EID", "depression"])
y = merged["depression"]

X = X.dropna(axis=1, how="all")

# Impute missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# 6. Logistic regression
clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(X_train, y_train)

# 7. Evaluate
pred_prob = clf.predict_proba(X_test)[:, 1]
pred_label = (pred_prob > 0.5).astype(int)

print("AUC (raw proteins):", roc_auc_score(y_test, pred_prob))
print("Accuracy (raw proteins):", accuracy_score(y_test, pred_label))