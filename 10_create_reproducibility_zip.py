# 10_create_reproducibility_zip.py
# AIGB 7290 — ESG Litigation Classifier
# Reproducibility zip for Step 10: Descriptive Analysis (word clouds + n-grams)
# Huang, Kiernan, Sooknanan | Fordham University
#
# Produces:
#   snapshots/snapshot_10_descriptive_analysis_v1_4232026.{csv,json}
#   step_zips/snapshot_10_descriptive_analysis_v1_4232026.zip
#     Contains: snapshot CSV+JSON, 10_manifest.json, script, all 11 PNGs,
#               ngrams CSV, config.py, CLAUDE.md, status.md,
#               reproducibility_manifest_v1_4232026.json (SHA-256 for every file)

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
DESC_DIR      = ESG_CORPUS_OUTPUTS / "descriptive_analysis"
NOW           = datetime.now(timezone.utc).isoformat()

SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
ZIPS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("Step 10 Reproducibility Zip — Descriptive Analysis")
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
# Frozen feature matrix hash (master anchor — from step 05)
# ---------------------------------------------------------------------------
frozen_name = f"feature_matrix_{PACKAGE_VERSION}_frozen.csv"
frozen_path = ESG_CORPUS_OUTPUTS / frozen_name
frozen_hash = sha256(frozen_path) if frozen_path.exists() else "NOT_FOUND"
print(f"\nFrozen anchor: {frozen_name}")
print(f"  SHA-256: {frozen_hash}")

# ---------------------------------------------------------------------------
# Snapshot 10 — n-gram table as the canonical tabular record
# ---------------------------------------------------------------------------
print("\n--- Snapshot 10: Descriptive Analysis ---")
ngram_csv = DESC_DIR / f"ngrams{VERSION_SUFFIX}.csv"
df10 = pd.read_csv(ngram_csv)

slug10     = f"snapshot_10_descriptive_analysis{VERSION_SUFFIX}"
s10_csv    = SNAPSHOTS_DIR / f"{slug10}.csv"
s10_json   = SNAPSHOTS_DIR / f"{slug10}.json"

df10.to_csv(s10_csv, index=False)
s10_csv_hash = sha256(s10_csv)

m10 = json.loads((ROOT / "10_manifest.json").read_text())

meta10 = {
    "step":             "10_descriptive_analysis",
    "description":      "Word clouds and top-20 unigram/bigram/trigram extraction per ESG pillar.",
    "script":           "10_esg_descriptive_analysis.py",
    "version":          PACKAGE_VERSION,
    "date_stamp":       PACKAGE_DATE,
    "generated_at":     NOW,
    "n_cases":          m10["n_cases"],
    "label_counts":     m10["label_counts"],
    "plots":            m10["plots"],
    "ngram_csv":        m10["ngram_csv"],
    "top5_unigrams":    m10["top5_unigrams_per_pillar"],
    "sha256_outputs":   m10["sha256"],
    "frozen_feature_matrix": frozen_name,
    "frozen_sha256":    frozen_hash,
    "rows":             len(df10),
    "columns":          list(df10.columns),
    "sha256_csv":       s10_csv_hash,
    "manifest":         "10_manifest.json",
    "decisions": [
        "Top-20 unigrams, bigrams, trigrams per pillar via CountVectorizer (min_df=2)",
        "Custom stop-word list: ENGLISH_STOP_WORDS + legal boilerplate + 'outcome'",
        "Token pattern: alphabetic tokens 2+ chars, no pure numbers",
        "Word clouds: max_words=150, collocations=False, [OUTCOME] tokens stripped",
        "Per-pillar n-gram images (1x3 panel each); composite 2x2 word cloud panel",
    ],
}
with open(s10_json, "w") as f:
    json.dump(meta10, f, indent=2)

print(f"  {slug10}.csv  ({len(df10)} rows)  sha256: {s10_csv_hash[:16]}...")

# ---------------------------------------------------------------------------
# Enumerate all outputs to include in the zip
# ---------------------------------------------------------------------------
png_files = sorted(DESC_DIR.glob("*.png"))
csv_file  = ngram_csv

zip_file_list = [
    (f"snapshot_10/{slug10}.csv",  s10_csv),
    (f"snapshot_10/{slug10}.json", s10_json),
    ("10_manifest.json",           ROOT / "10_manifest.json"),
    ("10_esg_descriptive_analysis.py", ROOT / "10_esg_descriptive_analysis.py"),
    (f"ngrams/{ngram_csv.name}",   csv_file),
    ("config.py",                  ROOT / "config.py"),
    ("CLAUDE.md",                  ROOT / "CLAUDE.md"),
    ("status.md",                  ROOT / "status.md"),
]
for png in png_files:
    zip_file_list.append((f"charts/{png.name}", png))

# ---------------------------------------------------------------------------
# SHA-256 map for reproducibility manifest
# ---------------------------------------------------------------------------
file_hashes = {arcname: sha256(src) for arcname, src in zip_file_list if Path(src).exists()}

rep_manifest = {
    "package_name":  f"{slug10}.zip",
    "version":       PACKAGE_VERSION,
    "date_stamp":    PACKAGE_DATE,
    "generated_at":  NOW,
    "step":          "10_descriptive_analysis",
    "frozen_feature_matrix": {
        "file":   frozen_name,
        "sha256": frozen_hash,
        "note":   (
            "SHA-256 anchor for the frozen training dataset. "
            "Re-hash this file to verify that all analyses derive from the exact "
            "same 444-case corpus without post-hoc modification."
        ),
    },
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
out_zip = ZIPS_DIR / f"{slug10}.zip"
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
print(f"\nSnapshot CSV : {s10_csv.name}")
print(f"Snapshot JSON: {s10_json.name}")
print(f"Zip          : {out_zip.name}  ({zip_mb:.1f} MB)")
print(f"  Contains   : {len(png_files)} PNGs + ngrams CSV + snapshot + script + manifests")
print(f"\nStep 10 reproducibility zip complete.")
