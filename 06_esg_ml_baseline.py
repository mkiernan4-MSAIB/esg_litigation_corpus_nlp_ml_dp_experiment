# 06_esg_ml_baseline.py
# AIGB 7290 — ESG Litigation Classifier
# Phase 5: Machine Learning Baseline + SHAP Interpretability
# Huang, Kiernan, Sooknanan | Fordham University
#
# Colab install:
#   !pip install xgboost shap scikit-learn pandas -q

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path resolution — local (config.py) or Google Colab fallback
# ---------------------------------------------------------------------------
try:
    from config import ROOT, ESG_CORPUS_OUTPUTS, ESG_CORPUS_LABELS_CSV
except ImportError:
    ROOT = Path("/content/drive/Shared Drives/ESG DL Project/esg_project")
    ESG_CORPUS_OUTPUTS  = ROOT / "esg_corpus_outputs"
    ESG_CORPUS_LABELS_CSV = ESG_CORPUS_OUTPUTS / "esg_corpus_labels.csv"

CLEANED_CSV  = ESG_CORPUS_OUTPUTS / "ESG_corpus_cleaned_v1.csv"
OUTPUTS_DIR  = ESG_CORPUS_OUTPUTS / "ml_baseline"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_OUT = ROOT / "06_manifest.json"

LABEL_ORDER = ["E", "S", "G", "Non-ESG"]

# ---------------------------------------------------------------------------
# Load and merge
# ---------------------------------------------------------------------------
print("Loading corpus...")
df_text   = pd.read_csv(CLEANED_CSV)
df_labels = pd.read_csv(ESG_CORPUS_LABELS_CSV)[["filename", "label"]]
df = df_text.merge(df_labels, on="filename", how="inner", suffixes=("_text", ""))

if "label_text" in df.columns:
    df.drop(columns=["label_text"], inplace=True)

df = df[df["label"].isin(LABEL_ORDER)].copy()
df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)
print(f"  Cases: {len(df)}")
print(df["label"].value_counts().to_string())

le = LabelEncoder()
le.fit(LABEL_ORDER)
y = le.transform(df["label"].values)

# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------
print("\nFitting TF-IDF (min_df=3, ngrams 1–2, max_features=20000)...")
vectorizer = TfidfVectorizer(min_df=3, ngram_range=(1, 2), sublinear_tf=True, max_features=20_000)
X = vectorizer.fit_transform(df["cleaned_text"])
print(f"  Vocabulary size: {X.shape[1]:,}")

# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------
classes  = np.unique(y)
cw_vals  = compute_class_weight("balanced", classes=classes, y=y)
cw_dict  = dict(zip(classes.tolist(), cw_vals.tolist()))
cw_named = dict(zip(LABEL_ORDER, cw_vals.round(3)))
print(f"  Class weights: {cw_named}")

# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def report(name, y_true, y_pred):
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    mcc      = matthews_corrcoef(y_true, y_pred)
    print(f"\n--- {name} ---")
    print(f"  Macro-F1 : {macro_f1:.4f}  |  MCC : {mcc:.4f}")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    return macro_f1, mcc

# ---------------------------------------------------------------------------
# Random Forest — 5-fold CV
# ---------------------------------------------------------------------------
print("\nTraining Random Forest (5-fold CV)...")
rf = RandomForestClassifier(n_estimators=500, class_weight=cw_dict, random_state=42, n_jobs=-1)
rf_pred = cross_val_predict(rf, X, y, cv=cv, n_jobs=-1)
rf_f1, rf_mcc = report("Random Forest", y, rf_pred)
rf.fit(X, y)  # refit on full corpus for SHAP and downstream use
joblib.dump(rf, OUTPUTS_DIR / "rf_model.pkl")
joblib.dump(vectorizer, OUTPUTS_DIR / "tfidf_vectorizer.pkl")

# ---------------------------------------------------------------------------
# XGBoost — 5-fold CV with sample weights
# ---------------------------------------------------------------------------
print("\nTraining XGBoost (5-fold CV)...")
sample_weights = np.array([cw_dict[c] for c in y])
xgb_clf = xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0,
)
xgb_pred_cv = np.zeros(len(y), dtype=int)
for train_idx, test_idx in cv.split(X, y):
    xgb_clf.fit(X[train_idx], y[train_idx], sample_weight=sample_weights[train_idx])
    xgb_pred_cv[test_idx] = xgb_clf.predict(X[test_idx])

xgb_f1, xgb_mcc = report("XGBoost", y, xgb_pred_cv)
xgb_clf.fit(X, y, sample_weight=sample_weights)  # refit on full corpus
joblib.dump(xgb_clf, OUTPUTS_DIR / "xgb_model.pkl")

# ---------------------------------------------------------------------------
# SHAP — Random Forest (TreeExplainer)
# ---------------------------------------------------------------------------
print("\nComputing SHAP values — Random Forest...")
feature_names = vectorizer.get_feature_names_out()
rng     = np.random.RandomState(42)
bg_idx  = rng.choice(X.shape[0], min(200, X.shape[0]), replace=False)
X_bg    = X[bg_idx].toarray()

explainer_rf = shap.TreeExplainer(rf)
sv_rf        = explainer_rf.shap_values(X_bg)  # list of arrays [n_classes, n_samples, n_features]

# Mean absolute SHAP across classes → top-20 global features
mean_abs = np.mean([np.abs(sv_rf[c]) for c in range(len(LABEL_ORDER))], axis=0).mean(axis=0)
top20    = np.argsort(mean_abs)[-20:][::-1]

fig, _ = plt.subplots(figsize=(11, 7))
shap.summary_plot(
    [sv_rf[c][:, top20] for c in range(len(LABEL_ORDER))],
    features=X_bg[:, top20],
    feature_names=feature_names[top20],
    class_names=LABEL_ORDER,
    plot_type="bar",
    show=False,
)
plt.title("SHAP Feature Importance — Random Forest (Top 20 Features)", fontsize=13)
plt.tight_layout()
fig.savefig(OUTPUTS_DIR / "shap_rf_beeswarm.png", dpi=200)
plt.close()
print(f"  Saved: shap_rf_beeswarm.png")

# Waterfall — highest-confidence Environmental case
e_idx    = list(LABEL_ORDER).index("E")
e_probs  = rf.predict_proba(X)[:, e_idx]
top_e    = int(np.argmax(e_probs))
sv_case  = explainer_rf.shap_values(X[top_e].toarray())[e_idx][0]
top20_loc = np.argsort(np.abs(sv_case))[-20:][::-1]

shap_exp = shap.Explanation(
    values=sv_case[top20_loc],
    base_values=explainer_rf.expected_value[e_idx],
    data=X[top_e].toarray()[0][top20_loc],
    feature_names=list(feature_names[top20_loc]),
)
fig2, _ = plt.subplots(figsize=(11, 7))
shap.waterfall_plot(shap_exp, show=False)
plt.title("SHAP Waterfall — Highest-Confidence Environmental Case (RF)", fontsize=12)
plt.tight_layout()
fig2.savefig(OUTPUTS_DIR / "shap_rf_waterfall_E.png", dpi=200)
plt.close()
print(f"  Saved: shap_rf_waterfall_E.png")

# ---------------------------------------------------------------------------
# SHAP — XGBoost (TreeExplainer)
# ---------------------------------------------------------------------------
print("\nComputing SHAP values — XGBoost...")
explainer_xgb = shap.TreeExplainer(xgb_clf)
sv_xgb        = explainer_xgb.shap_values(X_bg)

fig3, _ = plt.subplots(figsize=(11, 7))
if isinstance(sv_xgb, list):
    plot_data = [sv_xgb[c][:, top20] for c in range(len(LABEL_ORDER))]
else:
    plot_data = sv_xgb[:, top20]

shap.summary_plot(
    plot_data,
    features=X_bg[:, top20],
    feature_names=feature_names[top20],
    class_names=LABEL_ORDER if isinstance(sv_xgb, list) else None,
    plot_type="bar",
    show=False,
)
plt.title("SHAP Feature Importance — XGBoost (Top 20 Features)", fontsize=13)
plt.tight_layout()
fig3.savefig(OUTPUTS_DIR / "shap_xgb_beeswarm.png", dpi=200)
plt.close()
print(f"  Saved: shap_xgb_beeswarm.png")

# ---------------------------------------------------------------------------
# Save predictions and metrics
# ---------------------------------------------------------------------------
df["rf_pred"]  = le.inverse_transform(rf.predict(X))
df["xgb_pred"] = le.inverse_transform(xgb_clf.predict(X))
df.to_csv(OUTPUTS_DIR / "ml_baseline_predictions.csv", index=False)

metrics = {
    "random_forest": {"macro_f1": round(rf_f1, 4), "mcc": round(rf_mcc, 4)},
    "xgboost":       {"macro_f1": round(xgb_f1, 4), "mcc": round(xgb_mcc, 4)},
}
with open(OUTPUTS_DIR / "ml_baseline_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

manifest = {
    "script": "06_esg_ml_baseline.py",
    "n_cases": len(df),
    "tfidf_vocab_size": int(X.shape[1]),
    "rf_macro_f1":  round(rf_f1,  4),
    "rf_mcc":       round(rf_mcc, 4),
    "xgb_macro_f1": round(xgb_f1,  4),
    "xgb_mcc":      round(xgb_mcc, 4),
    "outputs_dir":  str(OUTPUTS_DIR),
    "shap_plots":   ["shap_rf_beeswarm.png", "shap_rf_waterfall_E.png", "shap_xgb_beeswarm.png"],
}
with open(MANIFEST_OUT, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nManifest written: {MANIFEST_OUT}")
print("Phase 5 — ML baseline complete.")
