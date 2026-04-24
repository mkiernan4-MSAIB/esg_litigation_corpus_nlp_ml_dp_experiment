# 10_esg_descriptive_analysis.py
# AIGB 7290 — ESG Litigation Classifier
# Phase: Descriptive Analysis — Word Clouds + N-Gram Extraction
# Huang, Kiernan, Sooknanan | Fordham University
#
# Outputs (versioned _v1_4232026):   11 PNGs total
#   esg_corpus_outputs/descriptive_analysis/
#     wordcloud_global_v1_4232026.png
#     wordcloud_E_v1_4232026.png
#     wordcloud_S_v1_4232026.png
#     wordcloud_G_v1_4232026.png
#     wordcloud_NonESG_v1_4232026.png
#     wordcloud_composite_pillars_v1_4232026.png   (2x2 panel)
#     ngrams_E_v1_4232026.png      (1x3: unigrams/bigrams/trigrams, top-20 each)
#     ngrams_S_v1_4232026.png
#     ngrams_G_v1_4232026.png
#     ngrams_NonESG_v1_4232026.png
#     ngrams_v1_4232026.csv        (machine-readable n-gram table)
#     10_manifest.json
#
# Note: matplotlib title strings use '--' (ASCII) not em-dash to avoid
# cp1252 encoding artifacts in PNG output on Windows.

import hashlib
import json
import re
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
try:
    from config import ROOT, ESG_CORPUS_OUTPUTS, PACKAGE_VERSION, PACKAGE_DATE, VERSION_SUFFIX
except ImportError:
    ROOT = Path("/content/drive/Shared Drives/ESG DL Project/esg_project")
    ESG_CORPUS_OUTPUTS = ROOT / "esg_corpus_outputs"
    PACKAGE_VERSION = "v1"; PACKAGE_DATE = "4232026"; VERSION_SUFFIX = "_v1_4232026"

CLEANED_CSV = ESG_CORPUS_OUTPUTS / "ESG_corpus_cleaned_v1.csv"
OUT_DIR     = ESG_CORPUS_OUTPUTS / "descriptive_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_ORDER  = ["E", "S", "G", "Non-ESG"]
LABEL_COLORS = {"E": "#2ca02c", "S": "#1f77b4", "G": "#ff7f0e", "Non-ESG": "#7f7f7f"}
LABEL_NAMES  = {"E": "Environmental", "S": "Social", "G": "Governance", "Non-ESG": "Non-ESG"}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()

# ---------------------------------------------------------------------------
# Stop words — standard English + legal boilerplate + masked token
# Preserve substantive legal vocabulary
# ---------------------------------------------------------------------------
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

CUSTOM_STOPS = set(ENGLISH_STOP_WORDS) | {
    # masked outcome token — must be excluded
    "outcome",
    # generic legal procedural boilerplate (non-substantive)
    "court", "courts", "plaintiff", "plaintiffs", "defendant", "defendants",
    "case", "cases", "claim", "claims", "action", "actions",
    "pursuant", "herein", "hereof", "thereof", "therein", "whereby",
    "said", "shall", "may", "also", "well", "one", "two", "three",
    "mr", "ms", "dr", "inc", "llc", "corp", "ltd", "co",
    "also", "however", "therefore", "thus", "further", "although",
    "january", "february", "march", "april", "june", "july",
    "august", "september", "october", "november", "december",
    "filed", "finding", "found", "held", "holding", "order",
    "judge", "judges", "justice", "opinion", "slip",
    "number", "nos", "no", "app", "supp", "cir", "dist",
    "id", "ibid", "supra", "infra", "et", "al", "see", "cf",
    # single characters and numbers handled by min_df / token pattern
}

# Token pattern: only real alphabetic tokens of 2+ chars, no pure numbers
TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z'\-]{1,}\b"

print("=" * 65)
print("ESG Descriptive Analysis — Word Clouds + N-Grams")
print(f"Version: {PACKAGE_VERSION}  |  Date: {PACKAGE_DATE}")
print("=" * 65)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("\nLoading corpus...")
df = pd.read_csv(CLEANED_CSV)
df = df[df["label"].isin(LABEL_ORDER)].copy()
df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)

# Strip [OUTCOME] tokens from text before all analysis
df["analysis_text"] = df["cleaned_text"].str.replace(r"\[OUTCOME\]", "", regex=True)
# Strip residual numbers and single chars
df["analysis_text"] = df["analysis_text"].str.replace(r"\b\d+\b", " ", regex=True)
df["analysis_text"] = df["analysis_text"].str.replace(r"\s+", " ", regex=True).str.strip()

print(f"  Cases loaded: {len(df)}")
print(df["label"].value_counts().to_string())

# ---------------------------------------------------------------------------
# Helper: fit vectorizer and get top-N n-grams for a corpus slice
# ---------------------------------------------------------------------------
def top_ngrams(texts, n_min, n_max, top_n=20):
    vec = CountVectorizer(
        ngram_range=(n_min, n_max),
        stop_words=list(CUSTOM_STOPS),
        token_pattern=TOKEN_PATTERN,
        min_df=2,
        max_features=50_000,
    )
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return []
    counts = np.asarray(X.sum(axis=0)).flatten()
    names  = vec.get_feature_names_out()
    pairs  = sorted(zip(names, counts), key=lambda x: -x[1])
    return [(t, int(c)) for t, c in pairs[:top_n]]

# ---------------------------------------------------------------------------
# N-Gram extraction — all pillars
# ---------------------------------------------------------------------------
print("\nExtracting n-grams...")
ngram_results = {}
for label in LABEL_ORDER:
    texts = df.loc[df["label"] == label, "analysis_text"].tolist()
    uni = top_ngrams(texts, 1, 1)
    bi  = top_ngrams(texts, 2, 2)
    tri = top_ngrams(texts, 3, 3)
    ngram_results[label] = {"unigrams": uni, "bigrams": bi, "trigrams": tri}
    print(f"  {LABEL_NAMES[label]:15s}  uni={len(uni)}  bi={len(bi)}  tri={len(tri)}")

# Save machine-readable CSV
rows = []
for label, ng in ngram_results.items():
    for rank, (token, count) in enumerate(ng["unigrams"], 1):
        rows.append({"label": label, "type": "unigram", "rank": rank, "token": token, "count": count})
    for rank, (token, count) in enumerate(ng["bigrams"], 1):
        rows.append({"label": label, "type": "bigram",  "rank": rank, "token": token, "count": count})
    for rank, (token, count) in enumerate(ng["trigrams"], 1):
        rows.append({"label": label, "type": "trigram", "rank": rank, "token": token, "count": count})

ngram_csv_path = OUT_DIR / f"ngrams{VERSION_SUFFIX}.csv"
pd.DataFrame(rows).to_csv(ngram_csv_path, index=False)
print(f"  N-gram CSV: {ngram_csv_path.name}")

# ---------------------------------------------------------------------------
# N-Gram visualization — one image per pillar (1 row × 3 cols: uni/bi/tri)
# ---------------------------------------------------------------------------
print("\nPlotting n-gram charts...")
ngram_pillar_paths = {}

for label in LABEL_ORDER:
    color = LABEL_COLORS[label]
    slug  = label.replace("-", "")
    fig, axes = plt.subplots(1, 3, figsize=(26, 12))
    fig.suptitle(
        f"Top 20 N-Grams -- {LABEL_NAMES[label]} Pillar\n"
        "AIGB 7290 | Fordham University | Huang, Kiernan, Sooknanan",
        fontsize=15, fontweight="bold", y=1.01,
    )

    for ax, (ng_type, ng_key) in zip(axes, [
        ("Unigrams",  "unigrams"),
        ("Bigrams",   "bigrams"),
        ("Trigrams",  "trigrams"),
    ]):
        pairs = ngram_results[label][ng_key]
        if not pairs:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(ng_type, fontsize=12, fontweight="bold")
            ax.axis("off")
            continue

        tokens = [p[0] for p in pairs][::-1]
        counts = [p[1] for p in pairs][::-1]

        bars = ax.barh(range(len(tokens)), counts, color=color, alpha=0.82, edgecolor="white", linewidth=0.4)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=9)
        ax.set_xlabel("Frequency", fontsize=9)
        ax.set_title(ng_type, fontsize=13, fontweight="bold", color=color, pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)

        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                    str(cnt), va="center", fontsize=7.5, color="#444444")

    fig.tight_layout()
    out_path = OUT_DIR / f"ngrams_{slug}{VERSION_SUFFIX}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    ngram_pillar_paths[label] = out_path
    print(f"  Saved: {out_path.name}")

# ---------------------------------------------------------------------------
# Word Cloud helper
# ---------------------------------------------------------------------------
def make_wordcloud(text: str, color: str, title: str, out_path: Path):
    wc = WordCloud(
        width=1600, height=900,
        background_color="white",
        stopwords=CUSTOM_STOPS,
        regexp=TOKEN_PATTERN,
        max_words=150,
        prefer_horizontal=0.75,
        collocations=False,
        color_func=lambda *a, **kw: color,
        min_font_size=9,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=18, fontweight="bold", pad=14, color="#222222")
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()

# ---------------------------------------------------------------------------
# Global word cloud
# ---------------------------------------------------------------------------
print("\nGenerating word clouds...")
global_text = " ".join(df["analysis_text"].tolist())
global_path = OUT_DIR / f"wordcloud_global{VERSION_SUFFIX}.png"
make_wordcloud(
    global_text, "#2d3a8c",
    "Global ESG Litigation Corpus -- Top Legal Terms\n"
    "AIGB 7290 | Fordham University | Huang, Kiernan, Sooknanan",
    global_path,
)
print(f"  Saved: {global_path.name}")

# ---------------------------------------------------------------------------
# Per-pillar word clouds
# ---------------------------------------------------------------------------
pillar_paths = {}
for label in LABEL_ORDER:
    pillar_text = " ".join(df.loc[df["label"] == label, "analysis_text"].tolist())
    slug = label.replace("-", "")
    out_path = OUT_DIR / f"wordcloud_{slug}{VERSION_SUFFIX}.png"
    n_cases  = int((df["label"] == label).sum())
    make_wordcloud(
        pillar_text, LABEL_COLORS[label],
        f"{LABEL_NAMES[label]} Pillar -- Top Legal Terms  (n={n_cases})\n"
        "AIGB 7290 | Fordham University | Huang, Kiernan, Sooknanan",
        out_path,
    )
    pillar_paths[label] = out_path
    print(f"  Saved: {out_path.name}")

# ---------------------------------------------------------------------------
# Composite 2×2 pillar word cloud panel
# ---------------------------------------------------------------------------
print("  Building composite panel...")
fig2, axes = plt.subplots(2, 2, figsize=(24, 14))
fig2.suptitle(
    "ESG Pillar Word Clouds -- Top Legal Terms by Classification\n"
    "AIGB 7290 | Fordham University | Huang, Kiernan, Sooknanan",
    fontsize=16, fontweight="bold",
)

for ax, label in zip(axes.flatten(), LABEL_ORDER):
    pillar_text = " ".join(df.loc[df["label"] == label, "analysis_text"].tolist())
    n_cases = int((df["label"] == label).sum())
    wc = WordCloud(
        width=1200, height=700, background_color="white",
        stopwords=CUSTOM_STOPS, regexp=TOKEN_PATTERN,
        max_words=100, prefer_horizontal=0.75, collocations=False,
        color_func=lambda *a, color=LABEL_COLORS[label], **kw: color,
        min_font_size=8,
    ).generate(pillar_text)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(
        f"{LABEL_NAMES[label]}  (n={n_cases})",
        fontsize=14, fontweight="bold", color=LABEL_COLORS[label], pad=10,
    )

fig2.tight_layout(rect=[0, 0, 1, 0.95])
composite_path = OUT_DIR / f"wordcloud_composite_pillars{VERSION_SUFFIX}.png"
fig2.savefig(composite_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Saved: {composite_path.name}")

# ---------------------------------------------------------------------------
# Top-5 unigrams integrity check — print to console
# ---------------------------------------------------------------------------
print("\n--- Integrity check: top 5 unigrams per pillar ---")
for label in LABEL_ORDER:
    top5 = [t for t, c in ngram_results[label]["unigrams"][:5]]
    print(f"  {LABEL_NAMES[label]:15s}: {top5}")

# ---------------------------------------------------------------------------
# Manifest + SHA-256
# ---------------------------------------------------------------------------
produced = sorted([f for f in OUT_DIR.iterdir() if f.suffix == ".png"])
hashes   = {f.name: sha256(f) for f in produced}
hashes[ngram_csv_path.name] = sha256(ngram_csv_path)

manifest = {
    "script":       "10_esg_descriptive_analysis.py",
    "version":      PACKAGE_VERSION,
    "date_stamp":   PACKAGE_DATE,
    "n_cases":      len(df),
    "label_counts": df["label"].value_counts().to_dict(),
    "outputs_dir":  str(OUT_DIR),
    "plots":        [f.name for f in produced],
    "ngram_csv":    ngram_csv_path.name,
    "sha256":       hashes,
    "top5_unigrams_per_pillar": {
        label: [t for t, c in ngram_results[label]["unigrams"][:5]]
        for label in LABEL_ORDER
    },
}
manifest_path = ROOT / "10_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nManifest written: {manifest_path.name}")
print(f"Outputs: {OUT_DIR}")
print(f"  {len(produced)} PNGs + ngrams CSV")
print("\nPhase: Descriptive Analysis complete.")
