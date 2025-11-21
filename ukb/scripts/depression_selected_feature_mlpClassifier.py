import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim

TOP_LIST = [20, 50, 100, 200, 500, 1000, 2000]

PROTEOMICS_FILE = "ukb/data/ukb_proteomics.tsv"
ICD_FILE = "ukb/UKB_ICD10F_diagnosis2.csv"
SIG_FILE = "significant_depression_proteins.tsv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    proteins = pd.read_csv(PROTEOMICS_FILE, sep="\t")
    proteins = proteins.rename(columns={"X.SampleID": "eid"})

    icd = pd.read_csv(ICD_FILE)
    icd["depression"] = (
        (icd["F32_diagnosis"] == True) |
        (icd["F33_diagnosis"] == True)
    ).astype(int)

    sig = pd.read_csv(SIG_FILE, sep="\t")
    return proteins, icd, sig


def prepare_features(proteins, icd, sig, topN):
    metadata_cols = ["eid", "batch", "fastingtime_0_0",
                     "bloodsample_date_0_0", "PlateID_0_0", "WellID_0_0"]

    real_protein_cols = [c for c in proteins.columns if c not in metadata_cols]

    selected = (
        sig.sort_values("effect_size", ascending=False)
           .head(topN)["protein"]
           .tolist()
    )
    selected = [p for p in selected if p in real_protein_cols]

    merged = proteins.merge(icd[["eid", "depression"]], on="eid", how="inner")
    X = merged[selected]
    y = merged["depression"]

    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    return X, y


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_test):
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    X_test_t = torch.tensor(X_test.values, dtype=torch.float32).to(DEVICE)

    # Compute class imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = n_neg / n_pos
    
    model = MLP(X_train.shape[1]).to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        logits = model.net[:-1](X_train_t)  # forward up to last layer
        logits = model.net[-1](logits)      # raw logits
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits_test = model.net[:-1](X_test_t)
        logits_test = model.net[-1](logits_test)
        pred_prob = torch.sigmoid(logits_test).cpu().numpy().flatten()

    pred = (pred_prob > 0.5).astype(int)
    return pred_prob, pred


def evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    prob, pred = train_mlp(X_train, y_train, X_test)
    auc = roc_auc_score(y_test, prob)
    acc = accuracy_score(y_test, pred)
    return auc, acc


proteins, icd, sig = load_data()

results = []

for topN in TOP_LIST:
    print(f"\nRunning Torch MLP for TOP {topN} proteins...")
    X, y = prepare_features(proteins, icd, sig, topN)
    auc, acc = evaluate(X, y)
    print(f"Top {topN}: AUC={auc:.4f}, Accuracy={acc:.4f}")
    results.append((topN, auc, acc))

print("\n==================== FINAL TORCH MLP RESULTS ====================")
print("TopN\tAUC\tAccuracy")
for topN, auc, acc in results:
    print(f"{topN}\t{auc:.4f}\t{acc:.4f}")
print("=================================================================")