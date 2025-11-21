import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load latent representations and ICD file
latents = pd.read_csv("results/latent_space/latent_space_full.tsv", sep="\t")
print("Latent space shape:", latents.shape)

# rename the tsv file from sample to eid
latents = latents.rename(columns={"sample":"eid"})

icd = pd.read_csv("UKB_ICD10F_diagnosis2.csv")
print("ICD file shape:", icd.shape)

# 2. Normalize column names
icd.columns = [c.strip().upper() for c in icd.columns]

# 3. Generate depression label: 1 if F32 or F33 diagnosis is TRUE
icd["depression"] = (
    icd["F32_DIAGNOSIS"].astype(str).str.upper().eq("TRUE")
    | icd["F33_DIAGNOSIS"].astype(str).str.upper().eq("TRUE")
).astype(int)

print(f"Depression prevalence: {icd['depression'].mean():.4f}")

# 4. Merge on participant ID
merged = latents.merge(icd[["EID", "depression"]], left_on="eid", right_on="EID", how="inner")
print("Merged shape:", merged.shape)

# 5. Prepare feature matrix and target vector
X = merged[[c for c in merged.columns if c.startswith("dim")]]
y = merged["depression"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 6. Train logistic regression classifier
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train, y_train)

# 7. Evaluate model performance
pred_prob = clf.predict_proba(X_test)[:, 1]
pred_label = (pred_prob > 0.5).astype(int)

auc = roc_auc_score(y_test, pred_prob)
acc = accuracy_score(y_test, pred_label)

print(f"AUC: {auc:.3f}")
print(f"Accuracy: {acc:.3f}")
print("\nClassification report:\n", classification_report(y_test, pred_label))

# 8. Plot predicted probability distributions
sns.histplot(pred_prob[y_test == 0], color="blue", label="Control", kde=True)
sns.histplot(pred_prob[y_test == 1], color="red", label="Depression", kde=True)
plt.title(f"Predicted probabilities (AUC = {auc:.3f})")
plt.xlabel("Predicted probability of depression")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("results/depression_prediction_auc.png", dpi=300)
plt.close()