# 09_create_reproducibility_package.py
# AIGB 7290 — ESG Litigation Classifier
# Reproducibility Package Generator
# Huang, Kiernan, Sooknanan | Fordham University
#
# Produces:
#   snapshots/                  — lightweight versioned CSV + JSON per pipeline step
#   reproducibility_package/    — full data, charts, notebook, scripts
#   ESG_Litigation_Classifier_reproducibility_v1.zip

import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from config import (
        ROOT, ESG_CORPUS_OUTPUTS, ESG_CORPUS_LABELS_CSV,
        ESG_CORPUS_FILTERED, ESG_CORPUS_PILLAR_METADATA_CSV,
        ESG_FILTER_LOG,
    )
except ImportError:
    ROOT = Path("/content/drive/Shared Drives/ESG DL Project/esg_project")
    ESG_CORPUS_OUTPUTS          = ROOT / "esg_corpus_outputs"
    ESG_CORPUS_LABELS_CSV       = ESG_CORPUS_OUTPUTS / "esg_corpus_labels.csv"
    ESG_CORPUS_FILTERED         = ROOT / "esg_corpus_filtered"
    ESG_CORPUS_PILLAR_METADATA_CSV = ESG_CORPUS_FILTERED / "esg_corpus_pillar_metadata.csv"
    ESG_FILTER_LOG              = ESG_CORPUS_FILTERED / "esg_filter_log.csv"

SNAPSHOTS_DIR = ROOT / "snapshots"
PKG_DIR       = ROOT / "reproducibility_package"
ZIP_PATH      = ROOT / "ESG_Litigation_Classifier_reproducibility_v1.zip"
ML_DIR        = ESG_CORPUS_OUTPUTS / "ml_baseline"
CLEANED_CSV   = ESG_CORPUS_OUTPUTS / "ESG_corpus_cleaned_v1.csv"
NOW           = datetime.now(timezone.utc).isoformat()

SNAPSHOTS_DIR.mkdir(exist_ok=True)
PKG_DIR.mkdir(exist_ok=True)
(PKG_DIR / "charts").mkdir(exist_ok=True)
(PKG_DIR / "data").mkdir(exist_ok=True)
(PKG_DIR / "manifests").mkdir(exist_ok=True)
(PKG_DIR / "scripts").mkdir(exist_ok=True)

print("=" * 65)
print("ESG Litigation Classifier — Reproducibility Package Generator")
print("=" * 65)

# ===========================================================================
# HELPER
# ===========================================================================
def write_snapshot(step_num, step_name, df, meta):
    slug    = f"snapshot_{step_num:02d}_{step_name}"
    csv_out = SNAPSHOTS_DIR / f"{slug}.csv"
    json_out= SNAPSHOTS_DIR / f"{slug}.json"
    df.to_csv(csv_out, index=False)
    meta["generated_at"] = NOW
    meta["rows"]         = len(df)
    meta["columns"]      = list(df.columns)
    with open(json_out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [{step_num:02d}] {slug}.csv  ({len(df)} rows) + .json")
    return csv_out, json_out

# ===========================================================================
# SNAPSHOT 01 — DEDUPLICATION
# ===========================================================================
print("\n--- Snapshot 01: Deduplication ---")
m1 = json.loads((ROOT / "01_manifest.json").read_text())

deduped_files  = sorted((ROOT / "esg_corpus_deduped").glob("*.md"))
dupe_files     = sorted((ROOT / "esg_corpus_deduped" / "esg_corpus_dupes").glob("*.md"))

rows = (
    [{"filename": f.name, "status": "canonical", "corpus": "esg_corpus_deduped"} for f in deduped_files] +
    [{"filename": f.name, "status": "duplicate",  "corpus": "esg_corpus_dupes"}  for f in dupe_files]
)
df01 = pd.DataFrame(rows)
meta01 = {
    "step":            "01_deduplication",
    "description":     "Cosine similarity deduplication (threshold > 0.95) of 1,282 raw Westlaw .md files.",
    "script":          "01_esg_deduplicate.py",
    "input_files":     m1["total_input"],
    "canonical_cases": m1["canonical_count"],
    "duplicates":      m1["duplicate_count"],
    "manifest":        "01_manifest.json",
    "decisions":       ["Cosine similarity threshold: 0.95", "Canonical = first-seen document", "364 duplicates quarantined in esg_corpus_dupes/"],
}
write_snapshot(1, "deduplication", df01, meta01)

# ===========================================================================
# SNAPSHOT 02 — NOISE FILTERING
# ===========================================================================
print("\n--- Snapshot 02: Noise Filtering ---")
m2   = json.loads((ROOT / "02_manifest.json").read_text())
df02 = pd.read_csv(ESG_FILTER_LOG)

meta02 = {
    "step":         "02_noise_filtering",
    "description":  "Keyword-signal noise filter retaining only ESG-relevant cases from the 918 deduplicated files.",
    "script":       "02_non_esg_filter_noise.py",
    "input_files":  918,
    "retained":     int((df02["decision"] == "retain").sum()) if "decision" in df02.columns else m2.get("retained", "see CSV"),
    "excluded":     int((df02["decision"] == "exclude").sum()) if "decision" in df02.columns else m2.get("excluded", "see CSV"),
    "manifest":     "02_manifest.json",
    "decisions":    ["ESG override threshold: 3 keyword hits", "Copyright-only cases excluded", "474 excluded to esg_corpus_excluded/"],
}
write_snapshot(2, "noise_filtering", df02, meta02)

# ===========================================================================
# SNAPSHOT 03 — CORPUS STATS
# ===========================================================================
print("\n--- Snapshot 03: Corpus Stats ---")
m3   = json.loads((ROOT / "03_manifest.json").read_text())
df03 = pd.read_csv(ESG_CORPUS_PILLAR_METADATA_CSV)

meta03 = {
    "step":        "03_corpus_stats",
    "description": "Per-file E/S/G/sustainability pillar signal scores for the 444 filtered cases.",
    "script":      "03_esg_corpus_stats.py",
    "input_files": 444,
    "manifest":    "03_manifest.json",
    "columns":     list(df03.columns),
    "decisions":   ["Pillar scores computed from keyword lexicon", "Signal strength = max(E, S, G, sus) scores"],
}
write_snapshot(3, "corpus_stats", df03, meta03)

# ===========================================================================
# SNAPSHOT 04 — LABEL CONSTRUCTION
# ===========================================================================
print("\n--- Snapshot 04: Label Construction ---")
m4   = json.loads((ROOT / "04_manifest.json").read_text())
df04 = pd.read_csv(ESG_CORPUS_LABELS_CSV)

dist = df04["label"].value_counts().to_dict()
meta04 = {
    "step":        "04_label_construction",
    "description": "Authoritative ESG labels: argmax(E, S, G pillar scores); tiebreaker G > E > S; zero-signal → Non-ESG.",
    "script":      "04_esg_label_construction.py",
    "n_cases":     len(df04),
    "class_distribution": dist,
    "is_sustainability_count": int(df04["is_sustainability"].sum()),
    "is_greenwash_count":      int(df04["is_greenwash"].sum()),
    "manifest":    "04_manifest.json",
    "decisions":   [
        "argmax(E, S, G) for primary label",
        "Tiebreaker: G > E > S",
        "Zero ESG signal → Non-ESG",
        "Sustainability = binary cross-cutting modifier, not primary class",
        "Tiebreaker date: April 22, 2026",
    ],
}
write_snapshot(4, "label_construction", df04, meta04)

# ===========================================================================
# SNAPSHOT 05 — TEXT CLEANING (summary only — full CSV is 24MB)
# ===========================================================================
print("\n--- Snapshot 05: Text Cleaning ---")
m5 = json.loads((ROOT / "05_manifest.json").read_text())

df_full = pd.read_csv(CLEANED_CSV, usecols=["filename", "label", "is_sustainability", "is_greenwash", "token_count", "exceeds_512"])

# Count [OUTCOME] masks per file from the full CSV
print("  Counting [OUTCOME] masks per file (reading full CSV)...")
df_text = pd.read_csv(CLEANED_CSV, usecols=["filename", "cleaned_text"])
df_text["outcome_mask_count"] = df_text["cleaned_text"].str.count(r"\[OUTCOME\]")
df05 = df_full.merge(df_text[["filename", "outcome_mask_count"]], on="filename")

meta05 = {
    "step":              "05_text_cleaning",
    "description":       "OCR normalization, Westlaw header removal, [OUTCOME] leakage masking, and token audit of 444 cases.",
    "script":            "05_esg_text_clean.py",
    "n_cases":           len(df05),
    "total_outcome_masks": int(df05["outcome_mask_count"].sum()),
    "files_with_masks":  int((df05["outcome_mask_count"] > 0).sum()),
    "files_no_masks":    int((df05["outcome_mask_count"] == 0).sum()),
    "pct_exceed_512":    round(df05["exceeds_512"].mean() * 100, 1),
    "mean_token_count":  round(df05["token_count"].mean(), 1),
    "max_token_count":   int(df05["token_count"].max()),
    "full_csv_path":     "esg_corpus_outputs/ESG_corpus_cleaned_v1.csv (24MB — excluded from git, included in zip)",
    "manifest":          "05_manifest.json",
    "decisions":         [
        "14 outcome-indicating terms masked as [OUTCOME]",
        "Context-aware regex + ALL-CAPS standalone pattern",
        "99.1% of cases exceed 512-token BERT limit",
        "Average 8,824 tokens/case → Longformer architecture required",
    ],
}
write_snapshot(5, "text_cleaning", df05, meta05)

# ===========================================================================
# SNAPSHOT 06 — ML BASELINE (metrics + predictions without full text)
# ===========================================================================
print("\n--- Snapshot 06: ML Baseline ---")
m6    = json.loads((ROOT / "06_manifest.json").read_text())
df_pred = pd.read_csv(ML_DIR / "ml_baseline_predictions.csv",
                      usecols=["filename", "label", "rf_pred", "xgb_pred", "token_count"])

meta06 = {
    "step":          "06_ml_baseline",
    "description":   "TF-IDF (min_df=3, ngrams 1-2) + Random Forest + XGBoost with class-weighted loss. 5-fold stratified CV.",
    "script":        "06_esg_ml_baseline.py",
    "n_cases":       m6["n_cases"],
    "tfidf_vocab":   m6["tfidf_vocab_size"],
    "random_forest": {"macro_f1": m6["rf_macro_f1"],  "mcc": m6["rf_mcc"]},
    "xgboost":       {"macro_f1": m6["xgb_macro_f1"], "mcc": m6["xgb_mcc"]},
    "manifest":      "06_manifest.json",
    "shap_plots":    m6["shap_plots"],
    "decisions":     [
        "class_weight='balanced' applied to RF",
        "Sample weights passed to XGBoost per-fold",
        "XGBoost is authoritative ML baseline (Macro-F1 0.8253)",
        "SHAP TreeExplainer — beeswarm + waterfall plots",
    ],
}
write_snapshot(6, "ml_baseline", df_pred, meta06)

# ===========================================================================
# JUPYTER NOTEBOOK
# ===========================================================================
print("\n--- Generating Jupyter notebook ---")

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code_cell(source, outputs=None):
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": outputs or [], "source": source,
    }

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": [

    md_cell(
        "# ESG Litigation Classifier — Reproducibility Notebook\n"
        "## AIGB 7290 Deep Learning | Fordham University\n"
        "### Huang, Kiernan, Sooknanan\n\n"
        "This notebook documents the complete preprocessing and modeling pipeline for classifying "
        "1,282 Westlaw litigation documents into Environmental (E), Social (S), Governance (G), "
        "and Non-ESG categories using NLP and deep learning. Each section corresponds to a versioned "
        "pipeline step with its canonical snapshot CSV and JSON metadata.\n\n"
        "**Corpus:** 1,282 raw OCR .md files → 918 unique → 444 high-signal → cleaned model input  \n"
        "**Models:** TF-IDF + XGBoost baseline; Longformer-base-4096 fine-tuning  \n"
        "**Explainability:** SHAP TreeExplainer (ML) + PartitionExplainer (Longformer)  "
    ),

    code_cell(
        "from pathlib import Path\nimport pandas as pd\nimport json\nimport matplotlib.pyplot as plt\n\n"
        "# Mount Google Drive if running in Colab\n"
        "try:\n"
        "    from google.colab import drive\n"
        "    drive.mount('/content/drive')\n"
        "    ROOT = Path('/content/drive/Shared Drives/ESG DL Project/esg_project')\n"
        "except ImportError:\n"
        "    ROOT = Path('.')  # local execution\n\n"
        "SNAPSHOTS = ROOT / 'snapshots'\n"
        "OUTPUTS   = ROOT / 'esg_corpus_outputs'\n"
        "print(f'Project root: {ROOT}')\n"
        "print(f'Snapshots dir exists: {SNAPSHOTS.exists()}')"
    ),

    md_cell(
        "## Step 01 — Deduplication\n\n"
        "1,282 raw Westlaw `.md` files were deduplicated using cosine similarity on TF-IDF vectors. "
        "Pairs with similarity > 0.95 were treated as duplicates; the first-seen document was retained "
        "as the canonical version. 364 duplicates were quarantined in `esg_corpus_dupes/`. "
        "918 unique cases were carried forward."
    ),

    code_cell(
        "df01 = pd.read_csv(SNAPSHOTS / 'snapshot_01_deduplication.csv')\n"
        "m01  = json.loads((SNAPSHOTS / 'snapshot_01_deduplication.json').read_text())\n"
        "print(f\"Input files  : {m01['input_files']}\")\n"
        "print(f\"Canonical    : {m01['canonical_cases']}\")\n"
        "print(f\"Duplicates   : {m01['duplicates']}\")\n"
        "df01['status'].value_counts()"
    ),

    md_cell(
        "## Step 02 — Noise Filtering\n\n"
        "The 918 deduplicated cases were passed through a keyword signal filter. Files with no ESG "
        "keyword hits — including 412 copyright-only Westlaw records — were excluded. Cases with "
        "weak ESG signal but fewer than 3 keyword hits were also excluded unless overridden by domain "
        "specificity. 444 high-signal cases were retained; 474 were quarantined in `esg_corpus_excluded/`."
    ),

    code_cell(
        "df02 = pd.read_csv(SNAPSHOTS / 'snapshot_02_noise_filtering.csv')\n"
        "m02  = json.loads((SNAPSHOTS / 'snapshot_02_noise_filtering.json').read_text())\n"
        "print(f\"Retained : {m02['retained']}\")\n"
        "print(f\"Excluded : {m02['excluded']}\")\n"
        "df02.head()"
    ),

    md_cell(
        "## Step 03 — Corpus Statistics & Pillar Metadata\n\n"
        "Per-file Environmental (E), Social (S), Governance (G), and sustainability signal scores "
        "were computed from a regulatory lexicon anchored in the GHG Protocol, UN Guiding Principles, "
        "ERISA § 404, and the UN Brundtland Commission (1987). These scores informed label construction "
        "in Step 04 and feature engineering in the descriptive analysis phase."
    ),

    code_cell(
        "df03 = pd.read_csv(SNAPSHOTS / 'snapshot_03_corpus_stats.csv')\n"
        "m03  = json.loads((SNAPSHOTS / 'snapshot_03_corpus_stats.json').read_text())\n"
        "print(f\"Cases: {len(df03)}\")\n"
        "df03[['E_score','S_score','G_score','sus_score']].describe().round(2)"
    ),

    md_cell(
        "## Step 04 — Label Construction\n\n"
        "Labels were assigned via `argmax(E_score, S_score, G_score)` with a deterministic tiebreaker "
        "(G > E > S). Cases with zero ESG signal across all pillars were assigned Non-ESG. A binary "
        "`is_sustainability` flag was appended per the UN Brundtland Commission (1987) definition — "
        "sustainability is a cross-cutting modifier, not a standalone primary class. "
        "Final distribution: G 171 (38.5%), Non-ESG 134 (30.2%), E 92 (20.7%), S 47 (10.6%)."
    ),

    code_cell(
        "df04 = pd.read_csv(SNAPSHOTS / 'snapshot_04_label_construction.csv')\n"
        "m04  = json.loads((SNAPSHOTS / 'snapshot_04_label_construction.json').read_text())\n"
        "print('Class distribution:')\n"
        "print(df04['label'].value_counts())\n"
        "print(f\"\\nis_sustainability: {df04['is_sustainability'].sum()} cases\")\n"
        "print(f\"is_greenwash     : {df04['is_greenwash'].sum()} case(s)\")"
    ),

    md_cell(
        "## Step 05 — Text Cleaning & Outcome Leakage Masking\n\n"
        "Each document underwent OCR normalization, removal of Westlaw caption boilerplate, and "
        "context-aware regex masking of 14 outcome-indicating terms (e.g., *dismissed*, *affirmed*, "
        "*remanded*) as `[OUTCOME]` tokens. An ALL-CAPS standalone pattern captured typographic "
        "variants. 3,028 masks were applied across 422 of 444 files; the 22 unmasked files are "
        "Non-ESG cases that contain no procedural outcome language. A token audit confirmed that "
        "99.1% of cases (440/444) exceed the 512-token limit of standard BERT-family models "
        "(mean: 8,824 tokens/case), necessitating the Longformer architecture."
    ),

    code_cell(
        "df05 = pd.read_csv(SNAPSHOTS / 'snapshot_05_text_cleaning.csv')\n"
        "m05  = json.loads((SNAPSHOTS / 'snapshot_05_text_cleaning.json').read_text())\n"
        "print(f\"Total [OUTCOME] masks    : {m05['total_outcome_masks']:,}\")\n"
        "print(f\"Files with masks         : {m05['files_with_masks']}\")\n"
        "print(f\"Files without masks      : {m05['files_no_masks']} (Non-ESG, expected)\")\n"
        "print(f\"Cases exceeding 512 tok  : {m05['pct_exceed_512']}%\")\n"
        "print(f\"Mean token count         : {m05['mean_token_count']:,}\")\n"
        "df05[['token_count','outcome_mask_count']].describe().round(1)"
    ),

    md_cell(
        "## Step 06 — Machine Learning Baseline\n\n"
        "TF-IDF vectorization (min_df=3, bigrams, 20,000 features) was applied to the masked text. "
        "A Random Forest and XGBoost classifier were trained with class-weighted loss to address the "
        "Social class underrepresentation (10.6% of corpus). Performance was evaluated via 5-fold "
        "stratified cross-validation using Macro-F1 and Matthews Correlation Coefficient.\n\n"
        "| Model | Macro-F1 | MCC |\n"
        "|---|---|---|\n"
        "| Random Forest | 0.6078 | 0.5721 |\n"
        "| **XGBoost** | **0.8253** | **0.8015** |\n\n"
        "XGBoost is the authoritative ML baseline. SHAP TreeExplainer attribution plots are shown below."
    ),

    code_cell(
        "df06 = pd.read_csv(SNAPSHOTS / 'snapshot_06_ml_baseline.csv')\n"
        "m06  = json.loads((SNAPSHOTS / 'snapshot_06_ml_baseline.json').read_text())\n"
        "print(f\"Random Forest  Macro-F1: {m06['random_forest']['macro_f1']}  MCC: {m06['random_forest']['mcc']}\")\n"
        "print(f\"XGBoost        Macro-F1: {m06['xgboost']['macro_f1']}  MCC: {m06['xgboost']['mcc']}\")\n"
        "print('\\nXGBoost prediction distribution:')\n"
        "df06['xgb_pred'].value_counts()"
    ),

    code_cell(
        "from IPython.display import Image, display\n"
        "charts = ROOT / 'esg_corpus_outputs' / 'ml_baseline'\n"
        "for fname in ['shap_rf_beeswarm.png', 'shap_rf_waterfall_E.png', 'shap_xgb_beeswarm.png']:\n"
        "    p = charts / fname\n"
        "    if p.exists():\n"
        "        print(f'\\n{fname}')\n"
        "        display(Image(filename=str(p), width=750))"
    ),

    md_cell(
        "## Step 07 — Longformer Fine-Tuning *(Colab GPU — run separately)*\n\n"
        "The Longformer architecture (`allenai/longformer-base-4096`) was selected as a direct "
        "consequence of the token audit finding: 99.1% of cases exceed the 512-token BERT limit. "
        "The model uses a sliding-window attention mechanism with global attention on the `[CLS]` "
        "classification token. The bottom 8 encoder layers are frozen to prevent catastrophic "
        "forgetting on the 444-case research corpus. Class-weighted cross-entropy loss penalizes "
        "errors on the underrepresented Social pillar and the single confirmed greenwashing case. "
        "Training is replicated across 3 random seeds with 5-fold cross-validation; results are "
        "reported as mean ± standard deviation of Macro-F1.\n\n"
        "*Execute `07_esg_longformer.py` in a Colab T4/A100 runtime. "
        "Then execute `08_esg_xai_visualizations.py` for t-SNE, attention heatmaps, and overlaid ROC curves.*"
    ),

    md_cell(
        "## References\n\n"
        "1. Blei et al. (2003). Latent Dirichlet Allocation. *JMLR*, 3, 993–1022.  \n"
        "2. Beltagy et al. (2020). Longformer: The Long-Document Transformer. *arXiv:2004.05150*.  \n"
        "3. Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.  \n"
        "4. UN Guiding Principles on Business and Human Rights (2011). OHCHR.  \n"
        "5. Uyghur Forced Labor Prevention Act, Pub. L. 117-78 (2021).  \n"
        "6. ERISA § 404, 29 U.S.C. § 1104 (fiduciary duties of prudence and loyalty).  \n"
        "7. DOL Final ESG Rule, 29 C.F.R. § 2550.404a-1 (2022).  \n"
        "8. UN Brundtland Commission. *Our Common Future* (1987). Oxford University Press.  "
    ),

    ],
}

nb_path = PKG_DIR / "ESG_Litigation_Classifier_Reproducibility.ipynb"
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"  Notebook: {nb_path.name}")

# ===========================================================================
# COPY ARTIFACTS INTO PACKAGE DIRECTORY
# ===========================================================================
print("\n--- Assembling reproducibility package ---")

# Charts
for png in (ML_DIR).glob("*.png"):
    shutil.copy2(png, PKG_DIR / "charts" / png.name)
    print(f"  chart: {png.name}")

# Data (small CSVs only — full CSVs via zip)
for csv_path in [ESG_CORPUS_LABELS_CSV]:
    shutil.copy2(csv_path, PKG_DIR / "data" / csv_path.name)
    print(f"  data: {csv_path.name}")

# All snapshots
for f in sorted(SNAPSHOTS_DIR.iterdir()):
    shutil.copy2(f, PKG_DIR / "data" / f.name)
print(f"  data: {len(list(SNAPSHOTS_DIR.iterdir()))} snapshot files")

# Manifests
for mf in sorted(ROOT.glob("[0-9][0-9]_manifest.json")):
    shutil.copy2(mf, PKG_DIR / "manifests" / mf.name)
    print(f"  manifest: {mf.name}")

# Scripts
for script in sorted(ROOT.glob("[0-9][0-9]_*.py")):
    shutil.copy2(script, PKG_DIR / "scripts" / script.name)
for f in ["config.py", "CLAUDE.md", "status.md"]:
    p = ROOT / f
    if p.exists():
        shutil.copy2(p, PKG_DIR / "scripts" / f)
print(f"  scripts: {len(list((PKG_DIR / 'scripts').iterdir()))} files")

# ===========================================================================
# BUILD ZIP
# ===========================================================================
print(f"\n--- Building zip: {ZIP_PATH.name} ---")

# Include full CSVs in zip (even though excluded from git)
extra_for_zip = [
    (CLEANED_CSV,                           "data/ESG_corpus_cleaned_v1.csv"),
    (ML_DIR / "ml_baseline_predictions.csv","data/ml_baseline_predictions.csv"),
    (ESG_CORPUS_PILLAR_METADATA_CSV,        "data/esg_corpus_pillar_metadata.csv"),
    (ROOT / "esg_corpus_filtered" / "esg_corpus_stats.txt", "data/esg_corpus_stats.txt"),
]

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for file in sorted(PKG_DIR.rglob("*")):
        if file.is_file():
            arcname = file.relative_to(PKG_DIR)
            zf.write(file, arcname)
    for src, arcname in extra_for_zip:
        if src.exists():
            zf.write(src, arcname)
            print(f"  + {arcname}")

zip_mb = ZIP_PATH.stat().st_size / 1e6
print(f"  Zip size: {zip_mb:.1f} MB -> {ZIP_PATH.name}")

# ===========================================================================
# MANIFEST FOR THIS SCRIPT
# ===========================================================================
manifest09 = {
    "script":          "09_create_reproducibility_package.py",
    "generated_at":    NOW,
    "snapshots_dir":   str(SNAPSHOTS_DIR),
    "package_dir":     str(PKG_DIR),
    "zip_path":        str(ZIP_PATH),
    "zip_size_mb":     round(zip_mb, 1),
    "snapshots": [f.name for f in sorted(SNAPSHOTS_DIR.iterdir()) if f.suffix in {".csv",".json"}],
    "notebook":        "ESG_Litigation_Classifier_Reproducibility.ipynb",
}
with open(ROOT / "09_manifest.json", "w") as f:
    json.dump(manifest09, f, indent=2)

print(f"\nManifest written: 09_manifest.json")
print("\nReproducibility package complete.")
print(f"  snapshots/               — {len(list(SNAPSHOTS_DIR.iterdir()))} files")
print(f"  reproducibility_package/ — {len(list(PKG_DIR.rglob('*')))} files")
print(f"  {ZIP_PATH.name}")
