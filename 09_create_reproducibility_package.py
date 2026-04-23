# 09_create_reproducibility_package.py
# AIGB 7290 — ESG Litigation Classifier
# Reproducibility Package Generator — v2 (versioned filenames + SHA-256 integrity)
# Huang, Kiernan, Sooknanan | Fordham University
#
# Versioning scheme: all snapshot files, zips, and frozen datasets carry
# _{VERSION}_{MDDYYYY} suffix (e.g., _v1_4232026). Increment VERSION in
# config.py when pipeline decisions change.
#
# SHA-256 hashes serve as digital fingerprints anchoring every file to an
# immutable state, enabling journal reviewers to verify that reported results
# derive from the exact data and code committed here.
#
# Produces:
#   snapshots/                         — versioned CSV + JSON per pipeline step
#   esg_corpus_outputs/
#     feature_matrix_v1_frozen.csv     — frozen training input with SHA-256 anchor
#   step_zips/
#     snapshot_01_deduplication_v1_4232026.zip
#     ...
#     snapshot_06_ml_baseline_v1_4232026.zip
#   Each zip contains:
#     - versioned snapshot CSV + JSON
#     - step manifest JSON
#     - step Python script(s)
#     - step-specific data/chart outputs
#     - reproducibility_manifest_v1_4232026.json  (SHA-256 for every file in zip)

import hashlib
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
        ESG_FILTER_LOG, PACKAGE_VERSION, PACKAGE_DATE, VERSION_SUFFIX,
    )
except ImportError:
    ROOT = Path("/content/drive/Shared Drives/ESG DL Project/esg_project")
    ESG_CORPUS_OUTPUTS             = ROOT / "esg_corpus_outputs"
    ESG_CORPUS_LABELS_CSV          = ESG_CORPUS_OUTPUTS / "esg_corpus_labels.csv"
    ESG_CORPUS_FILTERED            = ROOT / "esg_corpus_filtered"
    ESG_CORPUS_PILLAR_METADATA_CSV = ESG_CORPUS_FILTERED / "esg_corpus_pillar_metadata.csv"
    ESG_FILTER_LOG                 = ESG_CORPUS_FILTERED / "esg_filter_log.csv"
    PACKAGE_VERSION = "v1"
    PACKAGE_DATE    = "4232026"
    VERSION_SUFFIX  = f"_{PACKAGE_VERSION}_{PACKAGE_DATE}"

SNAPSHOTS_DIR = ROOT / "snapshots"
PKG_DIR       = ROOT / "reproducibility_package"
ZIPS_DIR      = ROOT / "step_zips"
ML_DIR        = ESG_CORPUS_OUTPUTS / "ml_baseline"
CLEANED_CSV   = ESG_CORPUS_OUTPUTS / "ESG_corpus_cleaned_v1.csv"
NOW           = datetime.now(timezone.utc).isoformat()

for d in [SNAPSHOTS_DIR, PKG_DIR, ZIPS_DIR,
          PKG_DIR / "charts", PKG_DIR / "data", PKG_DIR / "manifests", PKG_DIR / "scripts"]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("ESG Litigation Classifier — Reproducibility Package Generator")
print(f"Version: {PACKAGE_VERSION}  |  Date stamp: {PACKAGE_DATE}")
print("=" * 65)

# ===========================================================================
# SHA-256 UTILITY
# ===========================================================================
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def hash_dict(paths: list[tuple[str, Path]]) -> dict:
    """Return {arcname: sha256_hex} for all existing paths."""
    out = {}
    for arcname, p in paths:
        if Path(p).exists():
            out[arcname] = sha256(p)
    return out

# ===========================================================================
# FROZEN FEATURE MATRIX — SHA-256 anchor for training input
# ===========================================================================
print("\n--- Creating frozen feature matrix ---")
frozen_name = f"feature_matrix_{PACKAGE_VERSION}_frozen.csv"
frozen_path = ESG_CORPUS_OUTPUTS / frozen_name
shutil.copy2(CLEANED_CSV, frozen_path)
frozen_hash = sha256(frozen_path)
print(f"  {frozen_name}")
print(f"  SHA-256: {frozen_hash}")

# ===========================================================================
# SNAPSHOT HELPER — writes versioned CSV + JSON to snapshots/
# ===========================================================================
def write_snapshot(step_num: int, step_name: str, df: pd.DataFrame, meta: dict):
    slug     = f"snapshot_{step_num:02d}_{step_name}{VERSION_SUFFIX}"
    csv_out  = SNAPSHOTS_DIR / f"{slug}.csv"
    json_out = SNAPSHOTS_DIR / f"{slug}.json"
    df.to_csv(csv_out, index=False)
    meta.update({
        "version":      PACKAGE_VERSION,
        "date_stamp":   PACKAGE_DATE,
        "generated_at": NOW,
        "rows":         len(df),
        "columns":      list(df.columns),
        "sha256_csv":   sha256(csv_out),
    })
    with open(json_out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [{step_num:02d}] {slug}.csv  ({len(df)} rows)  sha256: {meta['sha256_csv'][:16]}...")
    return csv_out, json_out

# ===========================================================================
# SNAPSHOT 01 — DEDUPLICATION
# ===========================================================================
print("\n--- Snapshot 01: Deduplication ---")
m1 = json.loads((ROOT / "01_manifest.json").read_text())
deduped_files = sorted((ROOT / "esg_corpus_deduped").glob("*.md"))
dupe_files    = sorted((ROOT / "esg_corpus_deduped" / "esg_corpus_dupes").glob("*.md"))
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
    "decisions":       ["Cosine similarity threshold: 0.95", "364 duplicates quarantined in esg_corpus_dupes/"],
}
s01_csv, s01_json = write_snapshot(1, "deduplication", df01, meta01)

# ===========================================================================
# SNAPSHOT 02 — NOISE FILTERING
# ===========================================================================
print("\n--- Snapshot 02: Noise Filtering ---")
m2   = json.loads((ROOT / "02_manifest.json").read_text())
df02 = pd.read_csv(ESG_FILTER_LOG)
meta02 = {
    "step":        "02_noise_filtering",
    "description": "Keyword-signal noise filter retaining only ESG-relevant cases from the 918 deduplicated files.",
    "script":      "02_non_esg_filter_noise.py",
    "input_files": 918, "retained": int((df02["decision"] == "retain").sum()) if "decision" in df02.columns else "see CSV",
    "excluded":    int((df02["decision"] == "exclude").sum()) if "decision" in df02.columns else "see CSV",
    "manifest":    "02_manifest.json",
    "decisions":   ["ESG override threshold: 3 keyword hits", "474 excluded to esg_corpus_excluded/"],
}
s02_csv, s02_json = write_snapshot(2, "noise_filtering", df02, meta02)

# ===========================================================================
# SNAPSHOT 03 — CORPUS STATS
# ===========================================================================
print("\n--- Snapshot 03: Corpus Stats ---")
m3   = json.loads((ROOT / "03_manifest.json").read_text())
df03 = pd.read_csv(ESG_CORPUS_PILLAR_METADATA_CSV)
meta03 = {
    "step": "03_corpus_stats",
    "description": "Per-file E/S/G/sustainability pillar signal scores for the 444 filtered cases.",
    "script": "03_esg_corpus_stats.py", "input_files": 444, "manifest": "03_manifest.json",
    "decisions": ["Pillar scores from regulatory keyword lexicon"],
}
s03_csv, s03_json = write_snapshot(3, "corpus_stats", df03, meta03)

# ===========================================================================
# SNAPSHOT 04 — LABEL CONSTRUCTION
# ===========================================================================
print("\n--- Snapshot 04: Label Construction ---")
m4   = json.loads((ROOT / "04_manifest.json").read_text())
df04 = pd.read_csv(ESG_CORPUS_LABELS_CSV)
dist = df04["label"].value_counts().to_dict()
meta04 = {
    "step":        "04_label_construction",
    "description": "argmax(E, S, G pillar scores); tiebreaker G > E > S; zero-signal -> Non-ESG.",
    "script":      "04_esg_label_construction.py",
    "n_cases":     len(df04), "class_distribution": dist,
    "is_sustainability_count": int(df04["is_sustainability"].sum()),
    "is_greenwash_count":      int(df04["is_greenwash"].sum()),
    "manifest": "04_manifest.json",
    "decisions": ["argmax(E,S,G)", "Tiebreaker G > E > S", "Zero-signal -> Non-ESG",
                  "Sustainability = binary cross-cutting modifier"],
}
s04_csv, s04_json = write_snapshot(4, "label_construction", df04, meta04)

# ===========================================================================
# SNAPSHOT 05 — TEXT CLEANING (summary — full CSV is 24MB frozen separately)
# ===========================================================================
print("\n--- Snapshot 05: Text Cleaning ---")
m5     = json.loads((ROOT / "05_manifest.json").read_text())
df_sum = pd.read_csv(CLEANED_CSV, usecols=["filename", "label", "is_sustainability", "is_greenwash", "token_count", "exceeds_512"])
print("  Counting [OUTCOME] masks per file...")
df_txt = pd.read_csv(CLEANED_CSV, usecols=["filename", "cleaned_text"])
df_txt["outcome_mask_count"] = df_txt["cleaned_text"].str.count(r"\[OUTCOME\]")
df05   = df_sum.merge(df_txt[["filename", "outcome_mask_count"]], on="filename")
meta05 = {
    "step":              "05_text_cleaning",
    "description":       "[OUTCOME] leakage masking and token audit of 444 cases.",
    "script":            "05_esg_text_clean.py",
    "n_cases":           len(df05),
    "total_outcome_masks": int(df05["outcome_mask_count"].sum()),
    "files_with_masks":  int((df05["outcome_mask_count"] > 0).sum()),
    "pct_exceed_512":    round(df05["exceeds_512"].mean() * 100, 1),
    "mean_token_count":  round(df05["token_count"].mean(), 1),
    "frozen_feature_matrix": frozen_name,
    "frozen_sha256":     frozen_hash,
    "manifest":          "05_manifest.json",
    "decisions":         ["14 outcome terms masked as [OUTCOME]", "99.1% exceed 512 tokens",
                          "Mean 8,824 tokens/case -> Longformer required"],
}
s05_csv, s05_json = write_snapshot(5, "text_cleaning", df05, meta05)

# ===========================================================================
# SNAPSHOT 06 — ML BASELINE
# ===========================================================================
print("\n--- Snapshot 06: ML Baseline ---")
m6      = json.loads((ROOT / "06_manifest.json").read_text())
df_pred = pd.read_csv(ML_DIR / "ml_baseline_predictions.csv",
                      usecols=["filename", "label", "rf_pred", "xgb_pred", "token_count"])
meta06 = {
    "step":          "06_ml_baseline",
    "description":   "TF-IDF + Random Forest + XGBoost with class-weighted loss. 5-fold stratified CV.",
    "script":        "06_esg_ml_baseline.py",
    "n_cases":       m6["n_cases"], "tfidf_vocab": m6["tfidf_vocab_size"],
    "random_forest": {"macro_f1": m6["rf_macro_f1"],  "mcc": m6["rf_mcc"]},
    "xgboost":       {"macro_f1": m6["xgb_macro_f1"], "mcc": m6["xgb_mcc"]},
    "frozen_feature_matrix": frozen_name,
    "frozen_sha256":         frozen_hash,
    "manifest":      "06_manifest.json", "shap_plots": m6["shap_plots"],
    "decisions":     ["XGBoost is authoritative ML baseline (Macro-F1 0.8253)",
                      "SHAP TreeExplainer beeswarm + waterfall plots"],
}
s06_csv, s06_json = write_snapshot(6, "ml_baseline", df_pred, meta06)

# ===========================================================================
# JUPYTER NOTEBOOK (unchanged structure, versioned filename)
# ===========================================================================
print("\n--- Generating Jupyter notebook ---")

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code_cell(source, outputs=None):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": outputs or [], "source": source}

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": [
    md_cell(
        f"# ESG Litigation Classifier — Reproducibility Notebook {PACKAGE_VERSION.upper()}\n"
        "## AIGB 7290 Deep Learning | Fordham University\n"
        "### Huang, Kiernan, Sooknanan\n\n"
        f"**Package version:** {PACKAGE_VERSION} | **Date stamp:** {PACKAGE_DATE}  \n"
        f"**Frozen training input SHA-256:** `{frozen_hash}`  \n\n"
        "All snapshot files carry the version suffix `" + VERSION_SUFFIX + "`. "
        "The SHA-256 hash above anchors the frozen feature matrix to this exact data state. "
        "Re-running `hashlib.sha256()` on `" + frozen_name + "` must reproduce this hash to confirm integrity."
    ),
    code_cell(
        "import hashlib, json\nfrom pathlib import Path\nimport pandas as pd\n\n"
        "try:\n    from google.colab import drive\n    drive.mount('/content/drive')\n"
        "    ROOT = Path('/content/drive/Shared Drives/ESG DL Project/esg_project')\n"
        "except ImportError:\n    ROOT = Path('.')\n\n"
        "SNAPSHOTS = ROOT / 'snapshots'\nOUTPUTS   = ROOT / 'esg_corpus_outputs'\n"
        f"VER       = '{VERSION_SUFFIX}'\n\n"
        "# Verify frozen feature matrix integrity\n"
        "frozen = OUTPUTS / '" + frozen_name + "'\n"
        "h = hashlib.sha256()\n"
        "[h.update(c) for c in iter(lambda: open(frozen,'rb').read(65536), b'')]\n"
        f"assert h.hexdigest() == '{frozen_hash}', 'INTEGRITY CHECK FAILED'\n"
        "print(f'Integrity verified: {frozen.name}')\nprint(f'SHA-256: {h.hexdigest()}')"
    ),
    md_cell("## Step 01 — Deduplication\n\n1,282 raw Westlaw `.md` files deduplicated via cosine similarity > 0.95. 364 duplicates quarantined; 918 canonical cases retained."),
    code_cell(f"df01 = pd.read_csv(SNAPSHOTS / f'snapshot_01_deduplication{{VER}}.csv')\nm01  = json.loads((SNAPSHOTS / f'snapshot_01_deduplication{{VER}}.json').read_text())\nprint(f\"Canonical: {{m01['canonical_cases']}} | Duplicates: {{m01['duplicates']}}\")\ndf01['status'].value_counts()"),
    md_cell("## Step 02 — Noise Filtering\n\n918 deduplicated cases filtered to 444 high-signal ESG documents. 474 excluded (copyright-only and zero-ESG-signal)."),
    code_cell(f"df02 = pd.read_csv(SNAPSHOTS / f'snapshot_02_noise_filtering{{VER}}.csv')\nm02  = json.loads((SNAPSHOTS / f'snapshot_02_noise_filtering{{VER}}.json').read_text())\nprint(f\"Retained: {{m02['retained']}} | Excluded: {{m02['excluded']}}\")\ndf02.head()"),
    md_cell("## Step 03 — Corpus Stats & Pillar Metadata\n\nPer-file E/S/G/sustainability signal scores computed from regulatory keyword lexicon."),
    code_cell(f"df03 = pd.read_csv(SNAPSHOTS / f'snapshot_03_corpus_stats{{VER}}.csv')\ndf03[['E_score','S_score','G_score','sus_score']].describe().round(2)"),
    md_cell("## Step 04 — Label Construction\n\nargmax(E, S, G); tiebreaker G > E > S; zero-signal -> Non-ESG. Distribution: G 38.5%, Non-ESG 30.2%, E 20.7%, S 10.6%."),
    code_cell(f"df04 = pd.read_csv(SNAPSHOTS / f'snapshot_04_label_construction{{VER}}.csv')\nprint(df04['label'].value_counts())\nprint(f\"is_sustainability: {{df04['is_sustainability'].sum()}} | is_greenwash: {{df04['is_greenwash'].sum())}}\""),
    md_cell("## Step 05 — Text Cleaning & [OUTCOME] Masking\n\n3,028 masks across 422/444 files. 99.1% of cases exceed 512-token BERT limit (mean 8,824 tokens). Longformer required."),
    code_cell(f"df05 = pd.read_csv(SNAPSHOTS / f'snapshot_05_text_cleaning{{VER}}.csv')\nm05  = json.loads((SNAPSHOTS / f'snapshot_05_text_cleaning{{VER}}.json').read_text())\nprint(f\"[OUTCOME] masks: {{m05['total_outcome_masks']:,}} across {{m05['files_with_masks']}} files\")\nprint(f\"Exceeding 512 tokens: {{m05['pct_exceed_512']}}% | Mean tokens: {{m05['mean_token_count']:,}}\")"),
    md_cell("## Step 06 — ML Baseline\n\nTF-IDF (min_df=3, bigrams, 20k features) + RF + XGBoost. 5-fold CV, class-weighted loss.\n\n| Model | Macro-F1 | MCC |\n|---|---|---|\n| Random Forest | 0.6078 | 0.5721 |\n| **XGBoost** | **0.8253** | **0.8015** |"),
    code_cell(f"df06 = pd.read_csv(SNAPSHOTS / f'snapshot_06_ml_baseline{{VER}}.csv')\nm06  = json.loads((SNAPSHOTS / f'snapshot_06_ml_baseline{{VER}}.json').read_text())\nprint(f\"XGBoost Macro-F1: {{m06['xgboost']['macro_f1']}} | MCC: {{m06['xgboost']['mcc']}}\")\nprint(f\"Frozen SHA-256 confirmed in m06: {{m06['frozen_sha256'][:32]}}...\")"),
    code_cell(
        "from IPython.display import Image, display\n"
        "for fname in ['shap_rf_beeswarm.png','shap_rf_waterfall_E.png','shap_xgb_beeswarm.png']:\n"
        "    p = ROOT / 'esg_corpus_outputs' / 'ml_baseline' / fname\n"
        "    if p.exists():\n        print(fname)\n        display(Image(filename=str(p), width=750))"
    ),
    md_cell("## Step 07 — Longformer Fine-Tuning *(Colab GPU — run separately)*\n\n`allenai/longformer-base-4096`. Bottom 8 layers frozen. Class-weighted CE loss. lr=2e-5, 10% warmup. 3 seeds x 5-fold CV. Run `07_esg_longformer.py` in T4/A100 Colab runtime."),
    md_cell(
        "## References\n\n"
        "1. Blei et al. (2003). LDA. *JMLR* 3, 993–1022.  \n"
        "2. Beltagy et al. (2020). Longformer. *arXiv:2004.05150*.  \n"
        "3. Lundberg & Lee (2017). SHAP. *NeurIPS*.  \n"
        "4. UN Guiding Principles on Business and Human Rights (2011).  \n"
        "5. Uyghur Forced Labor Prevention Act, Pub. L. 117-78 (2021).  \n"
        "6. ERISA § 404, 29 U.S.C. § 1104.  \n"
        "7. DOL Final ESG Rule, 29 C.F.R. § 2550.404a-1 (2022).  \n"
        "8. UN Brundtland Commission. *Our Common Future* (1987).  "
    ),
    ],
}

nb_name = f"ESG_Litigation_Classifier_Reproducibility{VERSION_SUFFIX}.ipynb"
nb_path = PKG_DIR / nb_name
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"  Notebook: {nb_name}")

# ===========================================================================
# PER-STEP ZIPS — versioned filenames + SHA-256 manifest inside each zip
# ===========================================================================
print("\n--- Building per-step zips ---")

step_zip_defs = [
    {
        "slug":  f"snapshot_01_deduplication{VERSION_SUFFIX}",
        "step":  "01_deduplication",
        "files": [
            (f"snapshot_01_deduplication{VERSION_SUFFIX}.csv",  s01_csv),
            (f"snapshot_01_deduplication{VERSION_SUFFIX}.json", s01_json),
            ("01_manifest.json",   ROOT / "01_manifest.json"),
            ("01_esg_deduplicate.py", ROOT / "01_esg_deduplicate.py"),
            ("config.py",          ROOT / "config.py"),
        ],
    },
    {
        "slug":  f"snapshot_02_noise_filtering{VERSION_SUFFIX}",
        "step":  "02_noise_filtering",
        "files": [
            (f"snapshot_02_noise_filtering{VERSION_SUFFIX}.csv",  s02_csv),
            (f"snapshot_02_noise_filtering{VERSION_SUFFIX}.json", s02_json),
            ("02_manifest.json",          ROOT / "02_manifest.json"),
            ("02_non_esg_filter_noise.py",ROOT / "02_non_esg_filter_noise.py"),
            ("esg_filter_log.csv",        ESG_FILTER_LOG),
            ("config.py",                 ROOT / "config.py"),
        ],
    },
    {
        "slug":  f"snapshot_03_corpus_stats{VERSION_SUFFIX}",
        "step":  "03_corpus_stats",
        "files": [
            (f"snapshot_03_corpus_stats{VERSION_SUFFIX}.csv",  s03_csv),
            (f"snapshot_03_corpus_stats{VERSION_SUFFIX}.json", s03_json),
            ("03_manifest.json",          ROOT / "03_manifest.json"),
            ("03_esg_corpus_stats.py",    ROOT / "03_esg_corpus_stats.py"),
            ("esg_corpus_pillar_metadata.csv", ESG_CORPUS_PILLAR_METADATA_CSV),
            ("esg_corpus_stats.txt",      ROOT / "esg_corpus_filtered" / "esg_corpus_stats.txt"),
            ("config.py",                 ROOT / "config.py"),
        ],
    },
    {
        "slug":  f"snapshot_04_label_construction{VERSION_SUFFIX}",
        "step":  "04_label_construction",
        "files": [
            (f"snapshot_04_label_construction{VERSION_SUFFIX}.csv",  s04_csv),
            (f"snapshot_04_label_construction{VERSION_SUFFIX}.json", s04_json),
            ("04_manifest.json",            ROOT / "04_manifest.json"),
            ("04_esg_label_construction.py",ROOT / "04_esg_label_construction.py"),
            ("esg_corpus_labels.csv",       ESG_CORPUS_LABELS_CSV),
            ("config.py",                   ROOT / "config.py"),
        ],
    },
    {
        "slug":  f"snapshot_05_text_cleaning{VERSION_SUFFIX}",
        "step":  "05_text_cleaning",
        "files": [
            (f"snapshot_05_text_cleaning{VERSION_SUFFIX}.csv",  s05_csv),
            (f"snapshot_05_text_cleaning{VERSION_SUFFIX}.json", s05_json),
            ("05_manifest.json",      ROOT / "05_manifest.json"),
            ("05_esg_text_clean.py",  ROOT / "05_esg_text_clean.py"),
            (frozen_name,             frozen_path),       # frozen feature matrix + hash in manifest
            ("config.py",             ROOT / "config.py"),
        ],
    },
    {
        "slug":  f"snapshot_06_ml_baseline{VERSION_SUFFIX}",
        "step":  "06_ml_baseline",
        "files": [
            (f"snapshot_06_ml_baseline{VERSION_SUFFIX}.csv",  s06_csv),
            (f"snapshot_06_ml_baseline{VERSION_SUFFIX}.json", s06_json),
            ("06_manifest.json",         ROOT / "06_manifest.json"),
            ("06_esg_ml_baseline.py",    ROOT / "06_esg_ml_baseline.py"),
            ("ml_baseline_metrics.json", ML_DIR / "ml_baseline_metrics.json"),
            (frozen_name,                frozen_path),    # frozen feature matrix repeated for self-containment
            ("charts/shap_rf_beeswarm.png",   ML_DIR / "shap_rf_beeswarm.png"),
            ("charts/shap_rf_waterfall_E.png",ML_DIR / "shap_rf_waterfall_E.png"),
            ("charts/shap_xgb_beeswarm.png",  ML_DIR / "shap_xgb_beeswarm.png"),
            (nb_name,                    nb_path),
            ("config.py",                ROOT / "config.py"),
            ("CLAUDE.md",                ROOT / "CLAUDE.md"),
            ("status.md",                ROOT / "status.md"),
        ],
    },
]

zip_sizes  = {}
zip_hashes = {}   # outer hash of each zip file itself

for step in step_zip_defs:
    out_path = ZIPS_DIR / f"{step['slug']}.zip"

    # Build SHA-256 map for all files that will enter this zip
    file_hashes = {}
    for arcname, src in step["files"]:
        if Path(src).exists():
            file_hashes[arcname] = sha256(src)

    # reproducibility_manifest inside zip
    rep_manifest = {
        "package_name":     f"{step['slug']}.zip",
        "version":          PACKAGE_VERSION,
        "date_stamp":       PACKAGE_DATE,
        "generated_at":     NOW,
        "step":             step["step"],
        "frozen_feature_matrix": {
            "file":   frozen_name,
            "sha256": frozen_hash,
            "note":   (
                "SHA-256 anchor for the frozen training dataset. "
                "Re-hash this file to verify that the Random Forest, XGBoost, and "
                "Longformer models were trained on the exact same data without post-hoc modification."
            ),
        },
        "file_hashes": file_hashes,
        "integrity_note": (
            "To verify: for each file listed in file_hashes, compute SHA-256 and compare. "
            "Any mismatch indicates the file was modified after package generation."
        ),
    }

    manifest_arcname = f"reproducibility_manifest{VERSION_SUFFIX}.json"
    manifest_bytes   = json.dumps(rep_manifest, indent=2).encode("utf-8")

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Write all data files
        for arcname, src in step["files"]:
            if Path(src).exists():
                zf.write(src, arcname)
        # Write reproducibility manifest last
        zf.writestr(manifest_arcname, manifest_bytes)

    zip_hashes[step["slug"]] = sha256(out_path)
    mb = out_path.stat().st_size / 1e6
    zip_sizes[step["slug"]] = round(mb, 1)
    print(f"  {step['slug']}.zip  ({mb:.1f} MB)  sha256: {zip_hashes[step['slug']][:16]}...")

total_mb = sum(zip_sizes.values())
print(f"  Total: {total_mb:.1f} MB across {len(step_zip_defs)} zips")

# ===========================================================================
# MASTER MANIFEST — 09_manifest.json
# ===========================================================================
manifest09 = {
    "script":          "09_create_reproducibility_package.py",
    "version":         PACKAGE_VERSION,
    "date_stamp":      PACKAGE_DATE,
    "version_suffix":  VERSION_SUFFIX,
    "generated_at":    NOW,
    "frozen_feature_matrix": {
        "file":   frozen_name,
        "path":   str(frozen_path),
        "sha256": frozen_hash,
    },
    "step_zips":       {k: {"size_mb": v, "sha256": zip_hashes[k]} for k, v in zip_sizes.items()},
    "total_zip_mb":    round(total_mb, 1),
    "snapshots_dir":   str(SNAPSHOTS_DIR),
    "notebook":        nb_name,
    "integrity_note":  (
        "SHA-256 hashes in step_zips are the outer hashes of each zip file. "
        "SHA-256 in frozen_feature_matrix anchors the training dataset to an immutable state. "
        "Each zip also contains an internal reproducibility_manifest that hashes every file within it."
    ),
}
with open(ROOT / "09_manifest.json", "w") as f:
    json.dump(manifest09, f, indent=2)

print(f"\nMaster manifest written: 09_manifest.json")
print(f"Frozen feature matrix:   {frozen_name}")
print(f"  SHA-256: {frozen_hash}")
print(f"\nReproducibility package complete.")
print(f"  snapshots/  - {len(list(SNAPSHOTS_DIR.iterdir()))} files (versioned {VERSION_SUFFIX})")
print(f"  step_zips/  - {len(step_zip_defs)} zips, {total_mb:.1f} MB total")
print(f"  Each zip contains reproducibility_manifest{VERSION_SUFFIX}.json with SHA-256 for every file.")
