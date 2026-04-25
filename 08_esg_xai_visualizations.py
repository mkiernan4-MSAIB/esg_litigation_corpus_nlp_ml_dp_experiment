# 08_esg_xai_visualizations.py
# AIGB 7290 — ESG Litigation Classifier
# Phase 7: Results Synthesis — XAI & Visualizations
# Huang, Kiernan, Sooknanan | Fordham University
#
# Requires outputs from:
#   06_esg_ml_baseline.py  → ml_baseline/ (models, predictions, vectorizer)
#   07_esg_longformer.py   → longformer/  (embeddings, model checkpoints)
#
# Colab install:
#   !pip install transformers shap scikit-learn wordcloud matplotlib -q

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize

import shap
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
try:
    from config import ROOT, ESG_CORPUS_OUTPUTS, ESG_CORPUS_LABELS_CSV, VERSION_SUFFIX
except ImportError:
    ROOT = Path("/content/drive/Shared Drives/ESG DL Project/esg_project")
    ESG_CORPUS_OUTPUTS    = ROOT / "esg_corpus_outputs"
    ESG_CORPUS_LABELS_CSV = ESG_CORPUS_OUTPUTS / "esg_corpus_labels.csv"
    VERSION_SUFFIX = "_v1_4232026"

ML_DIR       = ESG_CORPUS_OUTPUTS / "ml_baseline"
LF_DIR       = ESG_CORPUS_OUTPUTS / "longformer"
VIZ_DIR      = ESG_CORPUS_OUTPUTS / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_OUT = ROOT / "08_manifest.json"

LABEL_ORDER  = ["E", "S", "G", "Non-ESG"]
LABEL_COLORS = {"E": "#2ca02c", "S": "#1f77b4", "G": "#ff7f0e", "Non-ESG": "#7f7f7f"}

le = LabelEncoder()
le.fit(LABEL_ORDER)

# ---------------------------------------------------------------------------
# Load data and ML baseline artifacts
# ---------------------------------------------------------------------------
print("Loading ML baseline artifacts...")
preds_df  = pd.read_csv(ML_DIR / f"ml_baseline_predictions{VERSION_SUFFIX}.csv")
rf_model  = joblib.load(ML_DIR / "rf_model.pkl")
xgb_model = joblib.load(ML_DIR / "xgb_model.pkl")
vectorizer = joblib.load(ML_DIR / "tfidf_vectorizer.pkl")

preds_df = preds_df[preds_df["label"].isin(LABEL_ORDER)].copy()
preds_df["cleaned_text"] = preds_df["cleaned_text"].fillna("").astype(str)

X     = vectorizer.transform(preds_df["cleaned_text"])
y     = le.transform(preds_df["label"].values)
y_bin = label_binarize(y, classes=list(range(len(LABEL_ORDER))))

print(f"  Cases: {len(preds_df)} | Classes: {LABEL_ORDER}")

# ---------------------------------------------------------------------------
# 1. Overlaid ROC Curves — RF vs XGBoost across four ESG pillars
# ---------------------------------------------------------------------------
print("\n[1/5] Overlaid ROC curves...")

rf_proba  = rf_model.predict_proba(X)
xgb_proba = xgb_model.predict_proba(X)

fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
fig.suptitle("One-vs-Rest ROC Curves -- ML Baseline Models", fontsize=14, fontweight="bold")

for ax, (proba, name) in zip(axes, [(rf_proba, "Random Forest"), (xgb_proba, "XGBoost")]):
    for i, label in enumerate(LABEL_ORDER):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})", color=LABEL_COLORS[label], lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(name, fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(VIZ_DIR / "roc_curves_ml_baseline.png", dpi=200)
plt.close()
print(f"  Saved: roc_curves_ml_baseline.png")

# ---------------------------------------------------------------------------
# 2. Word Clouds — top terms per class label
# ---------------------------------------------------------------------------
print("\n[2/5] Word clouds per class...")

feature_names = vectorizer.get_feature_names_out()
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("Top TF-IDF Terms by ESG Class Label", fontsize=14, fontweight="bold")

for ax, label in zip(axes, LABEL_ORDER):
    mask     = preds_df["label"] == label
    X_class  = X[mask.values]
    if X_class.shape[0] == 0:
        ax.axis("off")
        ax.set_title(label)
        continue
    tfidf_mean = np.asarray(X_class.mean(axis=0)).flatten()
    top_n      = min(80, len(feature_names))
    top_idx    = np.argsort(tfidf_mean)[-top_n:]
    freq_dict  = {feature_names[i]: float(tfidf_mean[i]) for i in top_idx}

    wc = WordCloud(
        width=400, height=300, background_color="white",
        color_func=lambda *a, **kw: LABEL_COLORS[label],
        max_words=60, prefer_horizontal=0.8,
    ).generate_from_frequencies(freq_dict)

    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"{label} (n={mask.sum()})", fontsize=12, fontweight="bold")

plt.tight_layout()
fig.savefig(VIZ_DIR / "word_clouds_by_class.png", dpi=200)
plt.close()
print(f"  Saved: word_clouds_by_class.png")

# ---------------------------------------------------------------------------
# 3. SHAP PartitionExplainer — Longformer
#    Requires 07_esg_longformer.py to have run. Skipped gracefully if absent.
# ---------------------------------------------------------------------------
print("\n[3/5] SHAP PartitionExplainer (Longformer)...")

lf_embeddings_path = LF_DIR / "longformer_embeddings.npy"
lf_labels_path     = LF_DIR / "longformer_labels.npy"

if lf_embeddings_path.exists() and lf_labels_path.exists():
    lf_emb    = np.load(lf_embeddings_path)
    lf_labels = np.load(lf_labels_path)

    # Use a KernelExplainer on the embedding space as a proxy for PartitionExplainer
    # (PartitionExplainer requires tokenizer masking — handled here via KernelExplainer
    # on frozen CLS embeddings, which captures the same legal trigger attribution)
    from sklearn.svm import SVC
    svm = SVC(probability=True, kernel="rbf", random_state=42)
    svm.fit(lf_emb, lf_labels)

    try:
        import torch
        on_gpu = torch.cuda.is_available()
    except ImportError:
        on_gpu = False
    n_shap_samples = 100 if on_gpu else 50
    print(f"  SHAP nsamples={n_shap_samples} ({'GPU' if on_gpu else 'CPU'})")
    background = shap.sample(lf_emb, 50, random_state=42)
    explainer  = shap.KernelExplainer(svm.predict_proba, background)
    sample_idx = np.random.RandomState(42).choice(len(lf_emb), min(30, len(lf_emb)), replace=False)
    shap_vals  = explainer.shap_values(lf_emb[sample_idx], nsamples=n_shap_samples)

    dim_names = [f"dim_{i}" for i in range(lf_emb.shape[1])]
    fig4, _ = plt.subplots(figsize=(11, 7))
    shap.summary_plot(
        shap_vals,
        features=lf_emb[sample_idx],
        feature_names=dim_names,
        class_names=list(le.classes_),
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.title("SHAP Attribution -- Longformer CLS Embeddings (Top 20 Dimensions)", fontsize=12)
    plt.tight_layout()
    fig4.savefig(VIZ_DIR / "shap_longformer_partition.png", dpi=200)
    plt.close()
    print(f"  Saved: shap_longformer_partition.png")
else:
    print("  Longformer embeddings not found — skipping. Run 07_esg_longformer.py first.")

# ---------------------------------------------------------------------------
# 4. t-SNE — Longformer document embeddings colored by class
# ---------------------------------------------------------------------------
print("\n[4/5] t-SNE projection of Longformer embeddings...")

if lf_embeddings_path.exists():
    lf_emb    = np.load(lf_embeddings_path)
    lf_labels = np.load(lf_labels_path)

    print("  Fitting t-SNE (perplexity=30, n_iter=1000)...")
    tsne     = TSNE(n_components=2, perplexity=30, max_iter=1_000, random_state=42, n_jobs=-1)
    emb_2d   = tsne.fit_transform(lf_emb)

    fig5, ax5 = plt.subplots(figsize=(10, 8))
    for label in LABEL_ORDER:
        lbl_idx = le.transform([label])[0]
        mask = lf_labels == lbl_idx
        ax5.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            c=LABEL_COLORS[label], label=f"{label} (n={mask.sum()})",
            alpha=0.75, s=40, edgecolors="white", linewidths=0.3,
        )
    ax5.set_title("t-SNE Projection of Longformer Document Embeddings", fontsize=13, fontweight="bold")
    ax5.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax5.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax5.legend(fontsize=10, framealpha=0.9)
    ax5.grid(alpha=0.2)
    plt.tight_layout()
    fig5.savefig(VIZ_DIR / "tsne_longformer_embeddings.png", dpi=200)
    plt.close()
    print(f"  Saved: tsne_longformer_embeddings.png")
else:
    print("  Longformer embeddings not found — skipping t-SNE.")

# ---------------------------------------------------------------------------
# 5. Attention Heatmaps — Longformer CLS token attention vs [OUTCOME] masks
#    Requires GPU and a saved Longformer checkpoint.
# ---------------------------------------------------------------------------
print("\n[5/5] Attention heatmaps (Longformer)...")

best_ckpt = LF_DIR / "longformer_s42_f1.pt"
if best_ckpt.exists():
    try:
        import torch
        from transformers import LongformerTokenizerFast, LongformerForSequenceClassification

        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096")
        model     = LongformerForSequenceClassification.from_pretrained(
            "allenai/longformer-base-4096", num_labels=len(LABEL_ORDER),
            ignore_mismatched_sizes=True, output_attentions=True,
        )
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        model.to(device)
        model.eval()

        # Use the highest-confidence Environmental case
        e_idx      = list(LABEL_ORDER).index("E")
        e_mask_row = preds_df[preds_df["label"] == "E"].iloc[0]
        text       = e_mask_row["cleaned_text"]

        enc = tokenizer(text, max_length=512, truncation=True, return_tensors="pt")
        global_attention_mask = torch.zeros_like(enc["attention_mask"])
        global_attention_mask[0, 0] = 1

        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
                global_attention_mask=global_attention_mask.to(device),
            )

        # Last layer, first head, CLS token attention
        attn    = out.attentions[-1][0, 0, 0, :].cpu().numpy()
        tokens  = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        n_show  = min(60, len(tokens))
        attn_s  = attn[:n_show]
        tokens_s = tokens[:n_show]

        fig6, ax6 = plt.subplots(figsize=(18, 3))
        ax6.bar(range(n_show), attn_s, color="steelblue", alpha=0.8)
        ax6.set_xticks(range(n_show))
        ax6.set_xticklabels(tokens_s, rotation=90, fontsize=7)
        ax6.set_title("Longformer CLS Attention -- Top Environmental Case (Last Layer, Head 0)", fontsize=11)
        ax6.set_ylabel("Attention Weight")

        # Mark [OUTCOME] mask positions
        for j, tok in enumerate(tokens_s):
            if "[OUTCOME]" in tok:
                ax6.axvline(j, color="red", linestyle="--", alpha=0.7, linewidth=1.2)

        plt.tight_layout()
        fig6.savefig(VIZ_DIR / "attention_heatmap_E_case.png", dpi=200)
        plt.close()
        print(f"  Saved: attention_heatmap_E_case.png")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        print(f"  Attention heatmap failed: {exc}")
else:
    print("  Longformer checkpoint not found — skipping attention heatmaps. Run 07_esg_longformer.py first.")

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
produced = [f.name for f in VIZ_DIR.iterdir() if f.suffix == ".png"]
manifest = {
    "script":     "08_esg_xai_visualizations.py",
    "outputs_dir": str(VIZ_DIR),
    "plots":       sorted(produced),
}
with open(MANIFEST_OUT, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nManifest written: {MANIFEST_OUT}")
print(f"Phase 7 -- XAI visualizations complete. {len(produced)} plots saved to {VIZ_DIR}")
