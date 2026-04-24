# 06_create_reproducibility_zip.py
# AIGB 7290 -- ESG Litigation Classifier
# Reproducibility zip for Step 06: ML Baseline (revised -- 27 plots)
# Huang, Kiernan, Sooknanan | Fordham University
#
# Produces:
#   snapshots/snapshot_06_ml_baseline_v1_4232026.{csv,json}  (regenerated from current outputs)
#   step_zips/snapshot_06_ml_baseline_v1_4232026.zip
#     Contains: predictions CSV + metrics JSON, all PNGs, .pkl models,
#               06_manifest.json, script, config.py, CLAUDE.md, status.md,
#               snapshot CSV+JSON, reproducibility_manifest_v1_4232026.json

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from config import (
        ROOT, ESG_CORPUS_OUTPUTS,
        PACKAGE_VERSION, PACKAGE_DATE, VERSION_SUFFIX,
    )
except ImportError:
    ROOT = Path("/content/drive/Shared Drives/ESG DL Project/esg_project")
    ESG_CORPUS_OUTPUTS = ROOT / "esg_corpus_outputs"
    PACKAGE_VERSION = "v1"; PACKAGE_DATE = "4232026"; VERSION_SUFFIX = "_v1_4232026"

SNAPSHOTS_DIR = ROOT / "snapshots"
ZIPS_DIR      = ROOT / "step_zips"
ML_DIR        = ESG_CORPUS_OUTPUTS / "ml_baseline"
NOW           = datetime.now(timezone.utc).isoformat()

SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
ZIPS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("Step 06 Reproducibility Zip -- ML Baseline (revised)")
print(f"Version: {PACKAGE_VERSION}  |  Date stamp: {PACKAGE_DATE}")
print("=" * 65)

# ---------------------------------------------------------------------------
# SHA-256 utility
# ---------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

# ---------------------------------------------------------------------------
# Frozen feature matrix hash (master anchor)
# ---------------------------------------------------------------------------
frozen_name = f"feature_matrix_{PACKAGE_VERSION}_frozen.csv"
frozen_path = ESG_CORPUS_OUTPUTS / frozen_name
frozen_hash = sha256(frozen_path) if frozen_path.exists() else "NOT_FOUND"
print(f"\nFrozen anchor: {frozen_name}")
print(f"  SHA-256: {frozen_hash}")
if frozen_hash != "NOT_FOUND":
    expected = "a2b95dfd616fc8573c1f27a4d4a4b18c02136c011a9ded6772b28bb75a00283e"
    status = "OK -- matches master anchor" if frozen_hash == expected else "MISMATCH -- investigate"
    print(f"  Integrity: {status}")

# ---------------------------------------------------------------------------
# Snapshot 06 -- predictions CSV as canonical tabular record
# ---------------------------------------------------------------------------
print("\n--- Snapshot 06: ML Baseline ---")
pred_csv  = ML_DIR / f"ml_baseline_predictions{VERSION_SUFFIX}.csv"
metrics_j = ML_DIR / f"ml_baseline_metrics{VERSION_SUFFIX}.json"

df06 = pd.read_csv(pred_csv)
slug06   = f"snapshot_06_ml_baseline{VERSION_SUFFIX}"
s06_csv  = SNAPSHOTS_DIR / f"{slug06}.csv"
s06_json = SNAPSHOTS_DIR / f"{slug06}.json"

df06.to_csv(s06_csv, index=False)
s06_csv_hash = sha256(s06_csv)

m06     = json.loads((ROOT / "06_manifest.json").read_text())
metrics = m06["metrics"]

meta06 = {
    "step":             "06_ml_baseline",
    "description":      "TF-IDF + Logistic Regression, Complement NB, Random Forest, XGBoost ML baseline. "
                        "Stratified 70/15/15 train/val/test split. SHAP Explanation-object API. "
                        "27 plots: 6 confusion matrices, 4 per-pillar ROC images, 2 feature importance, "
                        "2 global SHAP bar charts, 8 per-pillar dot beeswarms, 2 waterfall plots, "
                        "1 metrics table image.",
    "script":           "06_esg_ml_baseline.py",
    "version":          PACKAGE_VERSION,
    "date_stamp":       PACKAGE_DATE,
    "generated_at":     NOW,
    "n_cases":          m06["n_cases"],
    "split":            m06["split"],
    "tfidf_vocab_size": m06["tfidf_vocab_size"],
    "models":           m06["models"],
    "best_model":       m06["best_model"],
    "metrics_summary": {
        "majority_class":           {"macro_f1": metrics["majority_class"]["macro_f1"],         "mcc": metrics["majority_class"]["mcc"],         "auc": metrics["majority_class"]["auc_macro_ovr"]},
        "bow_logistic_regression":  {"macro_f1": metrics["bow_logistic_regression"]["macro_f1"],"mcc": metrics["bow_logistic_regression"]["mcc"],"auc": metrics["bow_logistic_regression"]["auc_macro_ovr"]},
        "logistic_regression":      {"macro_f1": metrics["logistic_regression"]["macro_f1"],    "mcc": metrics["logistic_regression"]["mcc"],    "auc": metrics["logistic_regression"]["auc_macro_ovr"]},
        "complement_naive_bayes":   {"macro_f1": metrics["complement_naive_bayes"]["macro_f1"], "mcc": metrics["complement_naive_bayes"]["mcc"], "auc": metrics["complement_naive_bayes"]["auc_macro_ovr"]},
        "random_forest":            {"macro_f1": metrics["random_forest"]["macro_f1"],          "mcc": metrics["random_forest"]["mcc"],          "auc": metrics["random_forest"]["auc_macro_ovr"]},
        "xgboost":                  {"macro_f1": metrics["xgboost"]["macro_f1"],               "mcc": metrics["xgboost"]["mcc"],               "auc": metrics["xgboost"]["auc_macro_ovr"]},
    },
    "plots":            m06["plots"],
    "frozen_feature_matrix": frozen_name,
    "frozen_sha256":    frozen_hash,
    "rows":             len(df06),
    "columns":          list(df06.columns),
    "sha256_csv":       s06_csv_hash,
    "sha256_outputs":   m06["sha256"],
    "manifest":         "06_manifest.json",
    "decisions": [
        "TF-IDF min_df=3, ngram_range=(1,2), max_features=20000",
        "Stratified 70/15/15 split: 310 train / 67 val / 67 test",
        "class_weight='balanced' for LR, RF; scale_pos_weight for XGBoost",
        "SHAP TreeExplainer -- Explanation-object API (explainer(pd.DataFrame(X, columns=fnames)))",
        "LabelEncoder sorts alphabetically; LE_LABEL_NAMES = [LABEL_NAMES[c] for c in le.classes_]",
        "All matplotlib titles use '--' (ASCII) to avoid cp1252 encoding artifacts on Windows",
        "[OUTCOME] masking verified -- no leakage in top 20 SHAP features",
    ],
}
with open(s06_json, "w") as f:
    json.dump(meta06, f, indent=2)

print(f"  {slug06}.csv  ({len(df06)} rows)  sha256: {s06_csv_hash[:16]}...")

# ---------------------------------------------------------------------------
# Enumerate all outputs to include in the zip
# ---------------------------------------------------------------------------
png_files = sorted(ML_DIR.glob("*.png"))
pkl_files = sorted(ML_DIR.glob("*.pkl"))

zip_file_list = [
    (f"snapshot_06/{slug06}.csv",  s06_csv),
    (f"snapshot_06/{slug06}.json", s06_json),
    ("06_manifest.json",           ROOT / "06_manifest.json"),
    ("06_esg_ml_baseline.py",      ROOT / "06_esg_ml_baseline.py"),
    (f"data/{pred_csv.name}",      pred_csv),
    (f"data/{metrics_j.name}",     metrics_j),
    ("config.py",                  ROOT / "config.py"),
    ("CLAUDE.md",                  ROOT / "CLAUDE.md"),
    ("status.md",                  ROOT / "status.md"),
]
for png in png_files:
    zip_file_list.append((f"charts/{png.name}", png))
for pkl in pkl_files:
    zip_file_list.append((f"models/{pkl.name}", pkl))

# ---------------------------------------------------------------------------
# SHA-256 map for reproducibility manifest
# ---------------------------------------------------------------------------
file_hashes = {arcname: sha256(Path(src)) for arcname, src in zip_file_list if Path(src).exists()}

rep_manifest = {
    "package_name":  f"{slug06}.zip",
    "version":       PACKAGE_VERSION,
    "date_stamp":    PACKAGE_DATE,
    "generated_at":  NOW,
    "step":          "06_ml_baseline",
    "best_model":    "xgboost",
    "best_metrics":  {"macro_f1": 0.7684, "mcc": 0.7467, "auc_macro_ovr": 0.9663},
    "frozen_feature_matrix": {
        "file":   frozen_name,
        "sha256": frozen_hash,
        "note":   (
            "SHA-256 anchor for the frozen training dataset. "
            "Re-hash this file to verify that all analyses derive from the exact "
            "444-case corpus without post-hoc modification."
        ),
    },
    "file_hashes":   file_hashes,
    "integrity_note": (
        "To verify: for each file in file_hashes, compute SHA-256 and compare. "
        "Any mismatch indicates the file was modified after package generation."
    ),
}

manifest_arcname = f"reproducibility_manifest{VERSION_SUFFIX}.json"
manifest_bytes   = json.dumps(rep_manifest, indent=2).encode("utf-8")

# ---------------------------------------------------------------------------
# Write zip
# ---------------------------------------------------------------------------
print("\n--- Building zip ---")
out_zip = ZIPS_DIR / f"{slug06}.zip"
missing = []
with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for arcname, src in zip_file_list:
        if Path(src).exists():
            zf.write(src, arcname)
        else:
            missing.append(str(src))
            print(f"  WARNING: missing {Path(src).name}")
    zf.writestr(manifest_arcname, manifest_bytes)

zip_hash = sha256(out_zip)
zip_mb   = out_zip.stat().st_size / 1e6

print(f"  {out_zip.name}")
print(f"  Size:    {zip_mb:.1f} MB")
print(f"  SHA-256: {zip_hash}")
if missing:
    print(f"  WARNING: {len(missing)} files missing from zip -- see above")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\nSnapshot CSV : {s06_csv.name}")
print(f"Snapshot JSON: {s06_json.name}")
print(f"Zip          : {out_zip.name}  ({zip_mb:.1f} MB)")
print(f"  PNGs       : {len(png_files)}")
print(f"  Models     : {len(pkl_files)} .pkl files")
print(f"  Data       : predictions CSV + metrics JSON")
print(f"  Frozen anchor: {frozen_hash[:16]}...")
print(f"\nStep 06 reproducibility zip complete.")
