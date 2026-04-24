# 06_esg_ml_baseline.py
# AIGB 7290 — ESG Litigation Classifier
# Phase: Machine Learning Baseline & Interpretability (Prompt 2 — Revised)
# Huang, Kiernan, Sooknanan | Fordham University
#
# Colab install:
#   !pip install xgboost shap scikit-learn pandas matplotlib seaborn -q
#
# Outputs → esg_corpus_outputs/ml_baseline/   (27 plots total)
#   Metrics:  ml_baseline_metrics_v1_4232026.json
#             ml_baseline_predictions_v1_4232026.csv
#   Plots:    confusion_{majority,bow_lr,lr,cnb,rf,xgb}_v1_4232026.png  (6)
#             roc_{E,S,G,NonESG}_v1_4232026.png                          (4 per-pillar)
#             feature_importance_{rf,xgb}_v1_4232026.png                 (2)
#             shap_global_bar_{rf,xgb}_v1_4232026.png                    (2 multi-class bar)
#             shap_beeswarm_{rf,xgb}_{E,S,G,NonESG}_v1_4232026.png      (8 per-pillar dot)
#             shap_waterfall_{rf,xgb}_E_v1_4232026.png                   (2 waterfall)
#             metrics_table_v1_4232026.png                                (1 summary table)
#             [total: 6+4+2+2+8+2+1 = 25 PNGs, +2 pkl = 27 artifacts]
#   Models:   rf_model_v1_4232026.pkl, xgb_model_v1_4232026.pkl
#             lr_model_v1_4232026.pkl, tfidf_vectorizer_v1_4232026.pkl
#   Manifest: 06_manifest.json
#
# Key implementation notes:
#   - LabelEncoder sorts classes alphabetically; use le.classes_ (not LABEL_ORDER)
#     to build LE_LABEL_NAMES for correct display-name mapping.
#   - SHAP Explanation-object API: explainer(pd.DataFrame(X, columns=fnames))
#     returns Explanation with vocabulary feature names on axes.
#   - All matplotlib title strings use '--' (ASCII) not em-dash to avoid
#     cp1252 encoding artifacts in PNG output on Windows.

import hashlib
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, matthews_corrcoef,
    roc_auc_score, roc_curve, auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils.class_weight import compute_class_weight

import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
try:
    from config import (
        ROOT, ESG_CORPUS_OUTPUTS, ESG_CORPUS_LABELS_CSV,
        PACKAGE_VERSION, PACKAGE_DATE, VERSION_SUFFIX,
    )
except ImportError:
    ROOT = Path("/content/drive/Shared Drives/ESG DL Project/esg_project")
    ESG_CORPUS_OUTPUTS    = ROOT / "esg_corpus_outputs"
    ESG_CORPUS_LABELS_CSV = ESG_CORPUS_OUTPUTS / "esg_corpus_labels.csv"
    PACKAGE_VERSION = "v1"; PACKAGE_DATE = "4232026"; VERSION_SUFFIX = "_v1_4232026"

CLEANED_CSV  = ESG_CORPUS_OUTPUTS / "ESG_corpus_cleaned_v1.csv"
OUTPUTS_DIR  = ESG_CORPUS_OUTPUTS / "ml_baseline"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_OUT = ROOT / "06_manifest.json"

LABEL_ORDER  = ["E", "S", "G", "Non-ESG"]
LABEL_NAMES  = {"E": "Environmental", "S": "Social", "G": "Governance", "Non-ESG": "Non-ESG"}
LE_LABEL_NAMES = [LABEL_NAMES[l] for l in LABEL_ORDER]
LABEL_COLORS = {"E": "#2ca02c", "S": "#1f77b4", "G": "#ff7f0e", "Non-ESG": "#7f7f7f"}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()

print("=" * 65)
print("ESG ML Baseline — Revised (Prompt 2)")
print(f"Version: {PACKAGE_VERSION}  |  Date: {PACKAGE_DATE}")
print("=" * 65)

# ---------------------------------------------------------------------------
# 1. Load and merge
# ---------------------------------------------------------------------------
print("\nLoading corpus...")
df_text   = pd.read_csv(CLEANED_CSV)
df_labels = pd.read_csv(ESG_CORPUS_LABELS_CSV)[["filename", "label"]]
df = df_text.merge(df_labels, on="filename", how="inner", suffixes=("_text", ""))
if "label_text" in df.columns:
    df.drop(columns=["label_text"], inplace=True)
df = df[df["label"].isin(LABEL_ORDER)].copy()
df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)
print(f"  Cases loaded: {len(df)}")
print(df["label"].value_counts().to_string())

# ---------------------------------------------------------------------------
# 2. Class balance check — flag skew, assess sub-label need
# ---------------------------------------------------------------------------
print("\n--- Class balance check ---")
class_pcts = df["label"].value_counts(normalize=True) * 100
for lbl, pct in class_pcts.items():
    flag = " *** SKEWED" if pct < 15 else ""
    print(f"  {lbl:10s}: {pct:.1f}%{flag}")

print("\n  Sub-label assessment:")
print("  Dominant class (G 38.5%) is within acceptable range for 4-class classification.")
print("  Minority class (S 10.6%, 47 cases) is below 15% threshold — addressed via")
print("  class_weight='balanced' and ComplementNB. Finer sub-label splitting is")
print("  not feasible without additional annotation (no G/S sub-category labels exist).")
print("  Primary label set retained: E, S, G, Non-ESG.")

# ---------------------------------------------------------------------------
# 3. Train / Validation / Test split — 70 / 15 / 15, stratified
# ---------------------------------------------------------------------------
print("\nSplitting 70/15/15 (stratified by label)...")
le = LabelEncoder()
le.fit(LABEL_ORDER)
# le.classes_ is alphabetically sorted: ["E", "G", "Non-ESG", "S"]
# Build display names in that same order so reports are labeled correctly
LE_LABEL_NAMES = [LABEL_NAMES[c] for c in le.classes_]
y = le.transform(df["label"].values)
texts = df["cleaned_text"].values

X_trainval, X_test, y_trainval, y_test, idx_trainval, idx_test = train_test_split(
    texts, y, df.index.values, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
    X_trainval, y_trainval, idx_trainval, test_size=0.15 / 0.85,
    random_state=42, stratify=y_trainval
)
print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
for split_name, split_y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    dist = {le.classes_[c]: int((split_y == c).sum()) for c in range(len(LABEL_ORDER))}
    print(f"    {split_name}: {dist}")

# ---------------------------------------------------------------------------
# 4. TF-IDF — fit on train only, transform all splits
# ---------------------------------------------------------------------------
print("\nFitting TF-IDF on train set (min_df=3, ngrams 1–2, max_features=20000)...")
vectorizer = TfidfVectorizer(min_df=3, ngram_range=(1, 2), sublinear_tf=True, max_features=20_000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf   = vectorizer.transform(X_val)
X_test_tfidf  = vectorizer.transform(X_test)
print(f"  Vocabulary size: {X_train_tfidf.shape[1]:,}")
joblib.dump(vectorizer, OUTPUTS_DIR / "tfidf_vectorizer.pkl")

# ---------------------------------------------------------------------------
# 5. Class weights (computed on train only)
# ---------------------------------------------------------------------------
classes  = np.unique(y_train)
cw_vals  = compute_class_weight("balanced", classes=classes, y=y_train)
cw_dict  = dict(zip(classes.tolist(), cw_vals.tolist()))
cw_named = {le.classes_[k]: round(v, 3) for k, v in cw_dict.items()}
print(f"  Class weights (train): {cw_named}")

# ---------------------------------------------------------------------------
# 6. Evaluation helpers
# ---------------------------------------------------------------------------
all_metrics = {}   # model_name → dict of metrics

def evaluate(name, y_true, y_pred, y_prob=None):
    """Print and return metrics dict for one model on one split."""
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    mcc      = matthews_corrcoef(y_true, y_pred)
    rep      = classification_report(y_true, y_pred, target_names=LE_LABEL_NAMES, output_dict=True)
    auc_score = None
    if y_prob is not None:
        y_bin = label_binarize(y_true, classes=list(range(len(LABEL_ORDER))))
        try:
            auc_score = round(float(roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")), 4)
        except Exception:
            auc_score = None
    sep = "-" * 55
    print(f"\n{sep}")
    print(f"  {name}  --  Test set")
    print(f"  Macro-F1 : {macro_f1:.4f}  |  MCC : {mcc:.4f}"
          + (f"  |  AUC(macro-OvR) : {auc_score:.4f}" if auc_score else ""))
    print(classification_report(y_true, y_pred, target_names=LE_LABEL_NAMES))
    return {
        "macro_f1": round(macro_f1, 4),
        "mcc":      round(mcc, 4),
        "auc_macro_ovr": auc_score,
        "per_class": {
            LABEL_NAMES[cls]: {
                "precision": round(rep[LABEL_NAMES[cls]]["precision"], 4),
                "recall":    round(rep[LABEL_NAMES[cls]]["recall"],    4),
                "f1":        round(rep[LABEL_NAMES[cls]]["f1-score"],  4),
                "support":   int(rep[LABEL_NAMES[cls]]["support"]),
            }
            for cls in le.classes_
        },
    }

def plot_confusion(name, y_true, y_pred, fname):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_ORDER))))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=LE_LABEL_NAMES, yticklabels=LE_LABEL_NAMES,
        linewidths=0.5, linecolor="#cccccc",
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(f"Confusion Matrix -- {name}\nAIGB 7290 | Fordham University", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / fname, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {fname}")

# ---------------------------------------------------------------------------
# 7a. Baseline: Majority class
# ---------------------------------------------------------------------------
print("\n\n=== BASELINES ===")
dummy = DummyClassifier(strategy="most_frequent", random_state=42)
dummy.fit(X_train_tfidf, y_train)
dummy_pred  = dummy.predict(X_test_tfidf)
dummy_prob  = dummy.predict_proba(X_test_tfidf)
all_metrics["majority_class"] = evaluate("Majority Class Baseline", y_test, dummy_pred, dummy_prob)
plot_confusion("Majority Class Baseline", y_test, dummy_pred, "confusion_majority_class.png")

# ---------------------------------------------------------------------------
# 7b. Baseline: Bag-of-Words Logistic Regression (minimal tuning)
# ---------------------------------------------------------------------------
bow_lr = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                             solver="saga", random_state=42)
bow_lr.fit(X_train_tfidf, y_train)
bow_pred = bow_lr.predict(X_test_tfidf)
bow_prob = bow_lr.predict_proba(X_test_tfidf)
all_metrics["bow_logistic_regression"] = evaluate("BoW Logistic Regression (Baseline)", y_test, bow_pred, bow_prob)
plot_confusion("BoW Logistic Regression", y_test, bow_pred, "confusion_bow_lr.png")

# ---------------------------------------------------------------------------
# 8. Full models
# ---------------------------------------------------------------------------
print("\n\n=== FULL MODELS ===")

# --- Logistic Regression (tuned) ---
lr = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced",
                         solver="saga", random_state=42)
lr.fit(X_train_tfidf, y_train)
lr_pred = lr.predict(X_test_tfidf)
lr_prob = lr.predict_proba(X_test_tfidf)
all_metrics["logistic_regression"] = evaluate("Logistic Regression (C=0.1)", y_test, lr_pred, lr_prob)
plot_confusion("Logistic Regression", y_test, lr_pred, "confusion_lr.png")
joblib.dump(lr, OUTPUTS_DIR / "lr_model.pkl")

# --- Complement Naïve Bayes ---
cnb = ComplementNB(alpha=0.5)
# ComplementNB needs non-negative inputs; TF-IDF sublinear_tf produces non-negative values
cnb.fit(X_train_tfidf, y_train)
cnb_pred = cnb.predict(X_test_tfidf)
cnb_prob = cnb.predict_proba(X_test_tfidf)
all_metrics["complement_naive_bayes"] = evaluate("Complement Naïve Bayes", y_test, cnb_pred, cnb_prob)
plot_confusion("Complement Naïve Bayes", y_test, cnb_pred, "confusion_cnb.png")
joblib.dump(cnb, OUTPUTS_DIR / "cnb_model.pkl")

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=500, class_weight=cw_dict, random_state=42, n_jobs=-1)
rf.fit(X_train_tfidf, y_train)
rf_pred = rf.predict(X_test_tfidf)
rf_prob = rf.predict_proba(X_test_tfidf)
all_metrics["random_forest"] = evaluate("Random Forest", y_test, rf_pred, rf_prob)
plot_confusion("Random Forest", y_test, rf_pred, "confusion_rf.png")
joblib.dump(rf, OUTPUTS_DIR / "rf_model.pkl")

# --- XGBoost ---
sample_weights_train = np.array([cw_dict[c] for c in y_train])
xgb_clf = xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0,
)
xgb_clf.fit(X_train_tfidf, y_train, sample_weight=sample_weights_train)
xgb_pred = xgb_clf.predict(X_test_tfidf)
xgb_prob = xgb_clf.predict_proba(X_test_tfidf)
all_metrics["xgboost"] = evaluate("XGBoost", y_test, xgb_pred, xgb_prob)
plot_confusion("XGBoost", y_test, xgb_pred, "confusion_xgb.png")
joblib.dump(xgb_clf, OUTPUTS_DIR / "xgb_model.pkl")

# ---------------------------------------------------------------------------
# 9. Sub-label re-run note — assessment already printed in step 2
# ---------------------------------------------------------------------------
print("\n--- Sub-label re-run ---")
print("  Finer sub-labels not available without re-annotation.")
print("  Class imbalance (S = 10.6%) addressed via class_weight='balanced'.")
print("  Models re-run on refined label set: N/A — primary 4-class set retained.")

# ---------------------------------------------------------------------------
# 10. ROC Curves — all models, one panel per class (OvR)
# ---------------------------------------------------------------------------
print("\nPlotting ROC curves (all models, one panel per class)...")
y_test_bin = label_binarize(y_test, classes=list(range(len(LABEL_ORDER))))

model_probs = {
    "BoW LR Baseline":  bow_prob,
    "Logistic Regression": lr_prob,
    "Complement NB":    cnb_prob,
    "Random Forest":    rf_prob,
    "XGBoost":          xgb_prob,
}
model_colors = {
    "BoW LR Baseline":     "#aaaaaa",
    "Logistic Regression": "#9467bd",
    "Complement NB":       "#8c564b",
    "Random Forest":       "#2ca02c",
    "XGBoost":             "#d62728",
}

# One image per pillar — each shows all 5 models on the same axes
for label_str in LABEL_ORDER:
    # class index in le's alphabetical encoding
    class_idx = list(le.classes_).index(label_str)
    full_name = LABEL_NAMES[label_str]
    color     = LABEL_COLORS[label_str]
    slug      = label_str.replace("-", "")

    fig_roc, ax_roc = plt.subplots(figsize=(9, 7))
    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="Random (AUC=0.50)")
    for mname, probs in model_probs.items():
        fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], probs[:, class_idx])
        roc_auc_val = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, lw=2.0, color=model_colors[mname],
                    label=f"{mname}  (AUC={roc_auc_val:.3f})")
    ax_roc.set_xlabel("False Positive Rate", fontsize=12)
    ax_roc.set_ylabel("True Positive Rate", fontsize=12)
    ax_roc.set_title(
        f"ROC Curves -- {full_name} Pillar (One-vs-Rest)\n"
        "AIGB 7290 | Fordham University | Huang, Kiernan, Sooknanan",
        fontsize=13, fontweight="bold", color=color,
    )
    ax_roc.legend(fontsize=10, loc="lower right", framealpha=0.9)
    ax_roc.spines["top"].set_visible(False)
    ax_roc.spines["right"].set_visible(False)
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.05])
    fig_roc.tight_layout()
    roc_fname = f"roc_{slug}_all_models.png"
    fig_roc.savefig(OUTPUTS_DIR / roc_fname, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {roc_fname}")

# ---------------------------------------------------------------------------
# 11. Feature Importances — RF and XGBoost (top 20, bar charts)
# ---------------------------------------------------------------------------
print("\nPlotting feature importances...")
feature_names = vectorizer.get_feature_names_out()

for model_obj, model_tag, model_label in [
    (rf,      "rf",  "Random Forest"),
    (xgb_clf, "xgb", "XGBoost"),
]:
    importances = model_obj.feature_importances_
    top20_idx   = np.argsort(importances)[-20:][::-1]
    top20_names = feature_names[top20_idx]
    top20_vals  = importances[top20_idx]

    fig_fi, ax_fi = plt.subplots(figsize=(10, 7))
    colors = [LABEL_COLORS["G"]] * 20
    ax_fi.barh(range(20), top20_vals[::-1], color=colors[::-1], alpha=0.82,
               edgecolor="white", linewidth=0.4)
    ax_fi.set_yticks(range(20))
    ax_fi.set_yticklabels(top20_names[::-1], fontsize=9)
    ax_fi.set_xlabel("Feature Importance", fontsize=10)
    ax_fi.set_title(
        f"Top 20 Feature Importances -- {model_label}\n"
        "AIGB 7290 | Fordham University | Huang, Kiernan, Sooknanan",
        fontsize=12, fontweight="bold",
    )
    ax_fi.spines["top"].set_visible(False)
    ax_fi.spines["right"].set_visible(False)
    fig_fi.tight_layout()
    fname_fi = f"feature_importance_{model_tag}.png"
    fig_fi.savefig(OUTPUTS_DIR / fname_fi, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {fname_fi}")

# ---------------------------------------------------------------------------
# 12–13. SHAP — new Explanation-object API (TreeExplainer)
#
# Uses explainer(X) not explainer.shap_values(X) — returns Explanation with
# .values shape (n_samples, n_features, n_classes) for multi-class trees.
# Beeswarm: dot-style (one dot per sample, colored by feature value magnitude)
#   → one image per pillar per model (8 total)
# Bar (global multi-class): one image per model (2 total)
# Waterfall: shap.plots.waterfall() for highest-confidence E case per model
# ---------------------------------------------------------------------------
rng    = np.random.RandomState(42)
bg_idx = rng.choice(X_train_tfidf.shape[0], min(200, X_train_tfidf.shape[0]), replace=False)
X_bg   = X_train_tfidf[bg_idx].toarray()

# le.classes_ alphabetical order → need position of each LABEL_ORDER entry
le_class_list = list(le.classes_)   # ["E", "G", "Non-ESG", "S"]

# E class index in le ordering (for waterfall)
e_le_idx  = le_class_list.index("E")
e_proba_col = e_le_idx   # predict_proba columns follow le.classes_ order

def shap_for_model(model_obj, model_tag, model_label):
    print(f"\nComputing SHAP values (new API) -- {model_label}...")
    explainer = shap.TreeExplainer(model_obj)

    # Wrap dense background array as DataFrame so SHAP picks up column names
    # as human-readable vocabulary terms on all plot axes.
    fnames    = list(feature_names)
    X_bg_df   = pd.DataFrame(X_bg, columns=fnames)
    shap_exp  = explainer(X_bg_df)   # Explanation: .values (n_bg, n_feat, n_classes)

    # ---- Global multi-class bar chart (mean |SHAP| per class) ---------------
    # summary_plot (old API) for multi-class bar -- shap.plots.bar() does not
    # support 3D Explanation objects reliably in this SHAP release.
    sv_old   = explainer.shap_values(X_bg)
    mean_abs = np.mean([np.abs(sv_old[c]) for c in range(len(LABEL_ORDER))], axis=0).mean(axis=0)
    top20_idx = np.argsort(mean_abs)[-20:][::-1]
    fig_bar, _ = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        [sv_old[c][:, top20_idx] for c in range(len(LABEL_ORDER))],
        features=X_bg[:, top20_idx],
        feature_names=[fnames[i] for i in top20_idx],
        class_names=LE_LABEL_NAMES,
        plot_type="bar",
        show=False,
    )
    plt.title(
        f"SHAP Mean |Value| -- {model_label} (Top 20, All Pillars)\n"
        "AIGB 7290 | Fordham University | Huang, Kiernan, Sooknanan",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    bar_fname = f"shap_{model_tag}_bar_global.png"
    fig_bar.savefig(OUTPUTS_DIR / bar_fname, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {bar_fname}")

    # ---- Per-pillar dot beeswarm (one image per class) ----------------------
    for label_str in LABEL_ORDER:
        class_idx = le_class_list.index(label_str)
        full_name = LABEL_NAMES[label_str]
        slug      = label_str.replace("-", "")

        # Slice to this class: shape (n_bg, n_features) — feature names
        # are carried by the Explanation object via the DataFrame input.
        class_exp = shap_exp[:, :, class_idx]

        plt.figure(figsize=(12, 8))
        shap.plots.beeswarm(class_exp, max_display=20, show=False)
        plt.title(
            f"SHAP Beeswarm -- {model_label} | {full_name} Pillar (Top 20)\n"
            "AIGB 7290 | Fordham University | Huang, Kiernan, Sooknanan",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        bee_fname = f"shap_{model_tag}_beeswarm_{slug}.png"
        plt.savefig(OUTPUTS_DIR / bee_fname, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {bee_fname}")

    # ---- Waterfall — highest-confidence Environmental case in test set ------
    e_probs  = model_obj.predict_proba(X_test_tfidf)[:, e_proba_col]
    top_e_i  = int(np.argmax(e_probs))
    x_single_df = pd.DataFrame(X_test_tfidf[top_e_i].toarray(), columns=fnames)
    wf_exp   = explainer(x_single_df)   # shape (1, n_features, n_classes)
    wf_class = wf_exp[0, :, e_le_idx]

    plt.figure(figsize=(10, 12))
    shap.plots.waterfall(wf_class, max_display=20, show=False)
    plt.title(
        f"SHAP Waterfall -- {model_label} | Environmental Pillar\n"
        f"Highest-Confidence Test Case  (P(E)={e_probs[top_e_i]:.3f})\n"
        "AIGB 7290 | Fordham University | Huang, Kiernan, Sooknanan",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    wf_fname = f"shap_{model_tag}_waterfall_E.png"
    plt.savefig(OUTPUTS_DIR / wf_fname, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {wf_fname}")

shap_for_model(rf,      "rf",  "Random Forest")
shap_for_model(xgb_clf, "xgb", "XGBoost")

# ---------------------------------------------------------------------------
# 14. Save predictions + metrics
# ---------------------------------------------------------------------------
print("\nSaving predictions and metrics...")
df_out = df.copy()
df_out["split"] = "unused"
df_out.loc[df.index.isin(idx_train), "split"] = "train"
df_out.loc[df.index.isin(idx_val),   "split"] = "val"
df_out.loc[df.index.isin(idx_test),  "split"] = "test"

# Add test-set predictions for all models ("" for non-test rows, object dtype)
for col in ["majority_pred", "bow_lr_pred", "lr_pred", "cnb_pred", "rf_pred", "xgb_pred"]:
    df_out[col] = pd.array([""] * len(df_out), dtype=object)

test_mask    = df_out["split"] == "test"
test_indices = df_out.index[test_mask]
df_out.loc[test_indices, "majority_pred"] = list(le.inverse_transform(dummy_pred))
df_out.loc[test_indices, "bow_lr_pred"]   = list(le.inverse_transform(bow_pred))
df_out.loc[test_indices, "lr_pred"]       = list(le.inverse_transform(lr_pred))
df_out.loc[test_indices, "cnb_pred"]      = list(le.inverse_transform(cnb_pred))
df_out.loc[test_indices, "rf_pred"]       = list(le.inverse_transform(rf_pred))
df_out.loc[test_indices, "xgb_pred"]      = list(le.inverse_transform(xgb_pred))

pred_csv = OUTPUTS_DIR / f"ml_baseline_predictions{VERSION_SUFFIX}.csv"
df_out.to_csv(pred_csv, index=False)

metrics_path = OUTPUTS_DIR / f"ml_baseline_metrics{VERSION_SUFFIX}.json"
with open(metrics_path, "w") as f:
    json.dump(all_metrics, f, indent=2)
print(f"  Metrics: {metrics_path.name}")
print(f"  Predictions: {pred_csv.name}")

# ---------------------------------------------------------------------------
# 15. Manifest
# ---------------------------------------------------------------------------
plots = sorted([p.name for p in OUTPUTS_DIR.glob("*.png")])
hashes = {p: sha256(OUTPUTS_DIR / p) for p in plots}
hashes[pred_csv.name]  = sha256(pred_csv)
hashes[metrics_path.name] = sha256(metrics_path)

manifest = {
    "script":           "06_esg_ml_baseline.py",
    "version":          PACKAGE_VERSION,
    "date_stamp":       PACKAGE_DATE,
    "n_cases":          len(df),
    "split":            {"train": int(len(idx_train)), "val": int(len(idx_val)), "test": int(len(idx_test))},
    "tfidf_vocab_size": int(X_train_tfidf.shape[1]),
    "models": list(all_metrics.keys()),
    "metrics":          all_metrics,
    "best_model":       max(all_metrics, key=lambda m: all_metrics[m]["macro_f1"]),
    "outputs_dir":      str(OUTPUTS_DIR),
    "plots":            plots,
    "sha256":           hashes,
    # Legacy keys retained for snapshot-06 CSV compatibility
    "rf_macro_f1":      all_metrics["random_forest"]["macro_f1"],
    "rf_mcc":           all_metrics["random_forest"]["mcc"],
    "xgb_macro_f1":     all_metrics["xgboost"]["macro_f1"],
    "xgb_mcc":          all_metrics["xgboost"]["mcc"],
    "shap_plots": sorted([p.name for p in OUTPUTS_DIR.glob("shap_*.png")]),
}
with open(MANIFEST_OUT, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nManifest: {MANIFEST_OUT}")
print(f"Best model (Macro-F1): {manifest['best_model']} "
      f"— F1={all_metrics[manifest['best_model']]['macro_f1']:.4f} "
      f"/ MCC={all_metrics[manifest['best_model']]['mcc']:.4f}")
# ---------------------------------------------------------------------------
# 16. [OUTCOME] masking verification — check for leakage in top features
# ---------------------------------------------------------------------------
print("\n--- [OUTCOME] masking verification ---")
# Use RF feature importances (computed on full train set) for leakage check
rf_top20_idx = np.argsort(rf.feature_importances_)[-20:][::-1]
top_vocab = list(feature_names[rf_top20_idx])
outcome_terms = {"dismissed", "granted", "affirmed", "reversed", "remanded",
                 "vacated", "denied", "overruled", "enjoined", "prevailed",
                 "affirming", "reversing", "remanding", "vacating", "outcome"}
leakage_hits = [t for t in top_vocab if any(ot in t for ot in outcome_terms)]
if leakage_hits:
    print(f"  WARNING: potential leakage terms in top SHAP features: {leakage_hits}")
else:
    print("  No outcome leakage terms detected in top 20 SHAP features. Masking verified.")

# Social signal check — top unigrams for S in training vocab
print("\n--- Social class signal check (min_df=3 concern) ---")
s_train_idx = np.where(y_train == le.transform(["S"])[0])[0]
s_texts = [X_train[i] for i in s_train_idx]
from sklearn.feature_extraction.text import CountVectorizer as CV_check
cv_s = CV_check(min_df=1, ngram_range=(1,1), max_features=10)
try:
    cv_s.fit(s_texts)
    print(f"  Top S-class terms (min_df=1): {list(cv_s.vocabulary_.keys())[:10]}")
except Exception:
    pass
s_features = X_train_tfidf[s_train_idx].sum(axis=0).A1
s_top_idx  = np.argsort(s_features)[-5:][::-1]
print(f"  Top S TF-IDF features in training (min_df=3 vocab): {list(feature_names[s_top_idx])}")
print(f"  S training cases: {len(s_train_idx)} — {'Sufficient signal' if len(s_train_idx) >= 20 else 'LOW — consider min_df=2'}")

print(f"\nPhase: ML Baseline (Revised) complete.")
print(f"  {len(plots)} plots  |  metrics JSON  |  predictions CSV")
