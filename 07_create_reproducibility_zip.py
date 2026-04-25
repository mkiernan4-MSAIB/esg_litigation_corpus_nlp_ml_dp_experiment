# 07_create_reproducibility_zip.py
# AIGB 7290 — ESG Litigation Classifier
# Reproducibility zip for Step 07: Longformer fine-tuning
# Huang, Kiernan, Sooknanan | Fordham University
#
# Produces:
#   snapshots/snapshot_07_longformer_v1_4232026.{csv,json}
#   step_zips/snapshot_07_longformer_v1_4232026.zip
#     Contains: embeddings.npy, labels.npy, 07_manifest.json, snapshot CSV+JSON,
#               script, config.py, CLAUDE.md, status.md,
#               reproducibility_manifest_v1_4232026.json (SHA-256 for every file)
#
# NOTE: Checkpoint files (longformer_s*_f*.pt) are 568 MB each (9 total = ~5.1 GB).
# They are excluded from the zip but their SHA-256 hashes are recorded in the
# reproducibility manifest so integrity can be verified without redistribution.

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from config import (
        ROOT, ESG_CORPUS_OUTPUTS,
        PACKAGE_VERSION, PACKAGE_DATE, VERSION_SUFFIX,
        ESG_LONGFORMER_DIR,
    )
except ImportError:
    ROOT = Path("/content/drive/Shared Drives/ESG DL Project/esg_project")
    ESG_CORPUS_OUTPUTS = ROOT / "esg_corpus_outputs"
    PACKAGE_VERSION = "v1"; PACKAGE_DATE = "4232026"; VERSION_SUFFIX = "_v1_4232026"
    ESG_LONGFORMER_DIR = ESG_CORPUS_OUTPUTS / "longformer"

SNAPSHOTS_DIR = ROOT / "snapshots"
ZIPS_DIR      = ROOT / "step_zips"
NOW           = datetime.now(timezone.utc).isoformat()

SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
ZIPS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("Step 07 Reproducibility Zip -- Longformer Fine-Tuning")
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
# Frozen feature matrix hash (master anchor -- from step 05)
# ---------------------------------------------------------------------------
frozen_name = f"feature_matrix_{PACKAGE_VERSION}_frozen.csv"
frozen_path = ESG_CORPUS_OUTPUTS / frozen_name
frozen_hash = sha256(frozen_path) if frozen_path.exists() else "NOT_FOUND"
print(f"\nFrozen anchor: {frozen_name}")
print(f"  SHA-256: {frozen_hash}")

# ---------------------------------------------------------------------------
# Checkpoint hashes (large files -- hash only, not included in zip)
# ---------------------------------------------------------------------------
print("\n--- Hashing checkpoint files (568 MB each -- excluded from zip) ---")
checkpoint_hashes = {}
seeds = [42, 123, 7]
for seed in seeds:
    for fold in [1, 2, 3]:
        ckpt = ESG_LONGFORMER_DIR / f"longformer_s{seed}_f{fold}.pt"
        if ckpt.exists():
            print(f"  Hashing {ckpt.name} ...", end=" ", flush=True)
            checkpoint_hashes[ckpt.name] = sha256(ckpt)
            print(checkpoint_hashes[ckpt.name][:16] + "...")
        else:
            print(f"  WARNING: {ckpt.name} NOT FOUND")
            checkpoint_hashes[ckpt.name] = "NOT_FOUND"

# ---------------------------------------------------------------------------
# Embedding / labels hashes
# ---------------------------------------------------------------------------
emb_path    = ESG_LONGFORMER_DIR / "longformer_embeddings.npy"
labels_path = ESG_LONGFORMER_DIR / "longformer_labels.npy"
emb_hash    = sha256(emb_path)    if emb_path.exists()    else "NOT_FOUND"
labels_hash = sha256(labels_path) if labels_path.exists() else "NOT_FOUND"

print(f"\nEmbeddings  : {emb_hash[:16]}...")
print(f"Labels      : {labels_hash[:16]}...")

# ---------------------------------------------------------------------------
# Snapshot 07 -- per-seed/fold results as tabular record
# ---------------------------------------------------------------------------
print("\n--- Snapshot 07: Longformer Results ---")
m07 = json.loads((ROOT / "07_manifest.json").read_text())

rows = []
for seed_str, result in m07["per_seed_results"].items():
    for fold_idx, f1 in enumerate(result["fold_f1s"], start=1):
        rows.append({
            "seed":       int(seed_str),
            "fold":       fold_idx,
            "macro_f1":   f1,
            "seed_mean":  result["mean"],
            "seed_std":   result["std"],
        })
df07 = pd.DataFrame(rows)

slug07   = f"snapshot_07_longformer{VERSION_SUFFIX}"
s07_csv  = SNAPSHOTS_DIR / f"{slug07}.csv"
s07_json = SNAPSHOTS_DIR / f"{slug07}.json"

df07.to_csv(s07_csv, index=False)
s07_csv_hash = sha256(s07_csv)

emb = np.load(emb_path) if emb_path.exists() else None

meta07 = {
    "step":                  "07_longformer",
    "description":           "Longformer-base-4096 3-seed x 3-fold cross-validation on 444-case ESG corpus.",
    "script":                "07_esg_longformer.py",
    "version":               PACKAGE_VERSION,
    "date_stamp":            PACKAGE_DATE,
    "generated_at":          NOW,
    "model":                 m07["model"],
    "max_len":               m07["max_len"],
    "seeds":                 m07["seeds"],
    "n_folds":               m07["n_folds"],
    "epochs":                m07["epochs"],
    "batch_size":            m07["batch_size"],
    "grad_accum":            m07["grad_accum"],
    "effective_batch":       m07["effective_batch"],
    "lr":                    m07["lr"],
    "freeze_layers":         m07["freeze_layers"],
    "use_amp":               m07["use_amp"],
    "completed_folds":       m07["completed_folds"],
    "per_seed_results":      m07["per_seed_results"],
    "global_macro_f1_mean":  m07["global_macro_f1_mean"],
    "global_macro_f1_std":   m07["global_macro_f1_std"],
    "embedding_shape":       list(emb.shape) if emb is not None else "NOT_FOUND",
    "sha256_embeddings":     emb_hash,
    "sha256_labels":         labels_hash,
    "sha256_checkpoints":    checkpoint_hashes,
    "checkpoint_note":       (
        "Checkpoints excluded from zip (568 MB each x 9 = ~5.1 GB). "
        "SHA-256 hashes recorded above for integrity verification."
    ),
    "frozen_feature_matrix": frozen_name,
    "frozen_sha256":         frozen_hash,
    "rows":                  len(df07),
    "columns":               list(df07.columns),
    "sha256_csv":            s07_csv_hash,
    "manifest":              "07_manifest.json",
    "decisions": [
        "Longformer required: 99.1% of 444 cases exceed 512-token BERT limit (avg 8,824 tokens/case)",
        "AMP FP16 mixed precision (GradScaler + autocast) for Colab T4 memory efficiency",
        "Gradient checkpointing enabled to reduce GPU VRAM usage",
        "Fold-level resume via 07_progress.json -- re-run same cell to continue interrupted session",
        "3 seeds x 3 folds x 5 epochs; freeze_layers=8 (Longformer attention layers frozen)",
        "Global Macro-F1: 0.4355 +/- 0.0370 -- underperforms XGBoost (0.7684) on 444-case corpus",
        "Underperformance expected: Longformer pre-trained on general text, fine-tuned on n=444",
        "CLS embeddings (shape 444 x 768) saved for downstream t-SNE / SHAP in script 08",
    ],
}
with open(s07_json, "w") as f:
    json.dump(meta07, f, indent=2)

print(f"  {slug07}.csv  ({len(df07)} rows)  sha256: {s07_csv_hash[:16]}...")

# ---------------------------------------------------------------------------
# Files to include in zip (checkpoints excluded -- too large)
# ---------------------------------------------------------------------------
zip_file_list = [
    (f"snapshot_07/{slug07}.csv",  s07_csv),
    (f"snapshot_07/{slug07}.json", s07_json),
    ("07_manifest.json",           ROOT / "07_manifest.json"),
    ("07_esg_longformer.py",       ROOT / "07_esg_longformer.py"),
    ("embeddings/longformer_embeddings.npy", emb_path),
    ("embeddings/longformer_labels.npy",     labels_path),
    ("config.py",                  ROOT / "config.py"),
    ("CLAUDE.md",                  ROOT / "CLAUDE.md"),
    ("status.md",                  ROOT / "status.md"),
]

# ---------------------------------------------------------------------------
# SHA-256 map for reproducibility manifest
# ---------------------------------------------------------------------------
file_hashes = {arcname: sha256(src) for arcname, src in zip_file_list if Path(src).exists()}
# Add checkpoint hashes even though files are not in zip
file_hashes["checkpoints_sha256_reference"] = checkpoint_hashes

rep_manifest = {
    "package_name":  f"{slug07}.zip",
    "version":       PACKAGE_VERSION,
    "date_stamp":    PACKAGE_DATE,
    "generated_at":  NOW,
    "step":          "07_longformer",
    "frozen_feature_matrix": {
        "file":   frozen_name,
        "sha256": frozen_hash,
        "note":   (
            "SHA-256 anchor for the frozen training dataset. "
            "Re-hash this file to verify that all analyses derive from the exact "
            "same 444-case corpus without post-hoc modification."
        ),
    },
    "checkpoint_note": (
        "The 9 Longformer checkpoint files (longformer_s{seed}_f{fold}.pt, 568 MB each) "
        "are excluded from this zip. Their SHA-256 hashes are recorded under "
        "'file_hashes.checkpoints_sha256_reference' for external verification."
    ),
    "file_hashes":   file_hashes,
    "integrity_note": (
        "To verify: for each file listed in file_hashes, compute SHA-256 and compare. "
        "Any mismatch indicates the file was modified after package generation."
    ),
}

manifest_arcname = f"reproducibility_manifest{VERSION_SUFFIX}.json"
manifest_bytes   = json.dumps(rep_manifest, indent=2).encode("utf-8")

# ---------------------------------------------------------------------------
# Write zip
# ---------------------------------------------------------------------------
print("\n--- Building zip ---")
out_zip = ZIPS_DIR / f"{slug07}.zip"
with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for arcname, src in zip_file_list:
        if Path(src).exists():
            zf.write(src, arcname)
        else:
            print(f"  WARNING: missing {src}")
    zf.writestr(manifest_arcname, manifest_bytes)

zip_hash = sha256(out_zip)
zip_mb   = out_zip.stat().st_size / 1e6

print(f"  {out_zip.name}")
print(f"  Size: {zip_mb:.1f} MB")
print(f"  SHA-256: {zip_hash}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\nSnapshot CSV : {s07_csv.name}")
print(f"Snapshot JSON: {s07_json.name}")
print(f"Zip          : {out_zip.name}  ({zip_mb:.1f} MB)")
print(f"  Contains   : embeddings.npy + labels.npy + snapshot + manifest + script")
print(f"  Checkpoints: {len(checkpoint_hashes)} hashes recorded (files excluded, 568 MB each)")
print(f"\nStep 07 reproducibility zip complete.")
print(f"\n--- SHA-256 summary (record in status.md / CLAUDE.md) ---")
print(f"  snapshot_07_longformer{VERSION_SUFFIX}.zip : {zip_hash}")
print(f"  longformer_embeddings.npy                  : {emb_hash}")
print(f"  longformer_labels.npy                      : {labels_hash}")
for name, h in checkpoint_hashes.items():
    print(f"  {name} : {h}")
