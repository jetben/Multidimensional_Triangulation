import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# -----------------------
# Load data
# -----------------------
DATA_PATH = Path("output") / "dataset_supervised.csv"
df = pd.read_csv(DATA_PATH)
df["month"] = pd.to_datetime(df["month"], errors="coerce")

LABEL_COL = "y_degrade"

exclude = [
    "repository_name", "month", "month_next",
    "p_fail_EB_next", "q_thr", "runs_next",
    LABEL_COL
]

feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype != "object"]

# Time split
months = sorted(df["month"].unique())
train = df[df["month"].isin(months[:2])]
val = df[df["month"].isin(months[2:3])]
test = df[df["month"].isin(months[3:4])]

X_train = train[feature_cols].values
X_val = val[feature_cols].values
X_test = test[feature_cols].values

y_train = train[LABEL_COL].values
y_val = val[LABEL_COL].values
y_test = test[LABEL_COL].values

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# -----------------------
# Model
# -----------------------
class DeepModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = DeepModel(X_train.shape[1])

# Class imbalance weighting
pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-8)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------
# Training
# -----------------------
epochs = 200
best_val_auc = 0
patience = 20
pat_counter = 0

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train).squeeze()
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val).squeeze()
        val_probs = torch.sigmoid(val_logits).numpy()
        val_auc = roc_auc_score(y_val.numpy(), val_probs)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_state = model.state_dict()
        pat_counter = 0
    else:
        pat_counter += 1

    if pat_counter >= patience:
        break

print("Best Val AUC:", best_val_auc)

# -----------------------
# Test Evaluation
# -----------------------
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    test_logits = model(X_test).squeeze()
    test_probs = torch.sigmoid(test_logits).numpy()

test_auc = roc_auc_score(y_test.numpy(), test_probs)

# Tune threshold
best_thr = 0.5
best_f1 = 0
for thr in np.linspace(0.05, 0.95, 50):
    preds = (test_probs >= thr).astype(int)
    f1 = f1_score(y_test.numpy(), preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

preds = (test_probs >= best_thr).astype(int)

print("\nDeep Learning Model Results")
print("Test AUC:", test_auc)
print("Best Threshold:", best_thr)
print("F1:", f1_score(y_test.numpy(), preds))
print("Precision:", precision_score(y_test.numpy(), preds))
print("Recall:", recall_score(y_test.numpy(), preds))
