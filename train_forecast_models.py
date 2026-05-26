"""
train_forecast_models.py  (FULL UPDATED)
---------------------------------------
Reads:  output/dataset_supervised.csv
Writes: output/forecast_results.csv

Fixes compared to previous version:
✅ Threshold tuning (on validation) for ALL models (LogReg, RF, MLP, heuristic)
✅ MLP uses sample_weight to handle imbalance (since MLPClassifier has no class_weight)
✅ Smaller, more regularized MLP (better for small train set like yours)
✅ Safer prints + saves

Run:
python train_forecast_models.py
"""

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


DATA_PATH = Path("output") / "dataset_supervised.csv"
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

LABEL_COL = "y_degrade"


# ----------------------------
# Metrics + threshold tuning
# ----------------------------
def evaluate(y_true, y_prob, thr=0.5) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(int)
    return {
        "AUC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Threshold": float(thr),
    }


def best_threshold_by_f1(y_true, y_prob) -> Tuple[float, float]:
    best_thr, best_f1 = 0.5, -1.0
    # Denser grid improves stability
    for thr in np.linspace(0.05, 0.95, 37):
        f1 = f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_thr, best_f1 = float(thr), float(f1)
    return best_thr, best_f1


def print_results(title: str, res: Dict[str, Dict[str, float]]):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    for model, m in res.items():
        print(f"{model:26s} | AUC={m['AUC']:.3f}  F1={m['F1']:.3f}  "
              f"P={m['Precision']:.3f}  R={m['Recall']:.3f}  thr={m['Threshold']:.2f}")


# ----------------------------
# Time split (your case: 4 months)
# ----------------------------
def time_split(df: pd.DataFrame, time_col="month"):
    """
    With 4 months, we do:
    Train = first 2 months
    Val   = 3rd month
    Test  = 4th month
    """
    months = sorted(df[time_col].dropna().unique())
    if len(months) < 4:
        raise ValueError(f"Need at least 4 months; found {len(months)}. Months={months}")

    train_months = set(months[:2])
    val_months = set(months[2:3])
    test_months = set(months[3:4])

    return (
        df[df[time_col].isin(train_months)].copy(),
        df[df[time_col].isin(val_months)].copy(),
        df[df[time_col].isin(test_months)].copy(),
        months,
    )


def get_numeric_feature_cols(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    ex = set(exclude)
    cols = []
    for c in df.columns:
        if c in ex:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


# ----------------------------
# Train + Evaluate (with val threshold tuning)
# ----------------------------
def train_all(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Dict[str, float]]:
    df_train, df_val, df_test, months = time_split(df)

    # Impute missing values using train medians
    med = df_train[feature_cols].median(numeric_only=True)
    for d in [df_train, df_val, df_test]:
        d[feature_cols] = d[feature_cols].fillna(med)

    # Scale for LR/MLP
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[feature_cols].values)
    X_val = scaler.transform(df_val[feature_cols].values)
    X_test = scaler.transform(df_test[feature_cols].values)

    y_train = df_train[LABEL_COL].astype(int).values
    y_val = df_val[LABEL_COL].astype(int).values
    y_test = df_test[LABEL_COL].astype(int).values

    results: Dict[str, Dict[str, float]] = {}

    print("\nMonths used:", months)
    print("Train rows:", len(df_train), "Val rows:", len(df_val), "Test rows:", len(df_test))
    print("Train positives:", int(y_train.sum()), "/", len(y_train))
    print("Val positives:", int(y_val.sum()), "/", len(y_val))
    print("Test positives:", int(y_test.sum()), "/", len(y_test))

    # ----------------------------
    # Baseline 1: heuristic using current EB prob (if present)
    # ----------------------------
    if "p_fail_EB" in df.columns:
        val_prob = df_val["p_fail_EB"].values
        best_thr, _ = best_threshold_by_f1(y_val, val_prob)
        test_prob = df_test["p_fail_EB"].values
        results["Heuristic(p_fail_EB)"] = evaluate(y_test, test_prob, thr=best_thr)

    # ----------------------------
    # Baseline 2: Logistic Regression (tuned threshold)
    # ----------------------------
    lr = LogisticRegression(max_iter=4000, class_weight="balanced", n_jobs=-1)
    lr.fit(X_train, y_train)

    val_prob = lr.predict_proba(X_val)[:, 1]
    best_thr, _ = best_threshold_by_f1(y_val, val_prob)

    test_prob = lr.predict_proba(X_test)[:, 1]
    results["LogReg(balanced)"] = evaluate(y_test, test_prob, thr=best_thr)

    # ----------------------------
    # Baseline 3: Random Forest (tuned threshold)
    # ----------------------------
    rf = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=5,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42
    )
    rf.fit(X_train, y_train)

    val_prob = rf.predict_proba(X_val)[:, 1]
    best_thr, _ = best_threshold_by_f1(y_val, val_prob)

    test_prob = rf.predict_proba(X_test)[:, 1]
    results["RandForest(balanced)"] = evaluate(y_test, test_prob, thr=best_thr)

    # ----------------------------
    # Deep model: MLP (smaller + regularized + sample_weight + tuned threshold)
    # ----------------------------
    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),   # smaller than (128,64) for your small training set
        activation="relu",
        alpha=1e-3,                    # stronger regularization
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=150,
        early_stopping=True,
        n_iter_no_change=12,
        random_state=42,
        verbose=False
    )

    # sample weights for imbalance (MLP has no class_weight)
    n_pos = max(int(y_train.sum()), 1)
    n_neg = max(len(y_train) - n_pos, 1)
    w_pos = n_neg / n_pos
    sample_weight = np.where(y_train == 1, w_pos, 1.0)

    mlp.fit(X_train, y_train, sample_weight=sample_weight)

    val_prob = mlp.predict_proba(X_val)[:, 1]
    best_thr, _ = best_threshold_by_f1(y_val, val_prob)

    test_prob = mlp.predict_proba(X_test)[:, 1]
    results["MLP(32-16, weighted)"] = evaluate(y_test, test_prob, thr=best_thr)

    return results


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH)
    df["month"] = pd.to_datetime(df["month"], errors="coerce")

    # Exclude IDs & leakage columns
    exclude_cols = [
        "repository_name", "month", "month_next",
        "p_fail_EB_next", "q_thr", "runs_next",
        LABEL_COL
    ]

    # Use all numeric features (exec + adoption + discourse if present)
    feature_cols = get_numeric_feature_cols(df, exclude_cols)

    if not feature_cols:
        raise ValueError("No numeric feature columns found. Check your dataset_supervised.csv columns.")

    print("Using feature columns:", feature_cols)

    results = train_all(df, feature_cols)
    print_results("Forecasting results (time-based split, tuned thresholds)", results)

    # Save results to CSV (paper-ready)
    out_rows = []
    for model, m in results.items():
        out_rows.append({"Model": model, **m})
    out_df = pd.DataFrame(out_rows).sort_values(by="AUC", ascending=False)
    out_df.to_csv(OUT_DIR / "forecast_results.csv", index=False)
    print("\nSaved:", (OUT_DIR / "forecast_results.csv").resolve())


if __name__ == "__main__":
    main()
