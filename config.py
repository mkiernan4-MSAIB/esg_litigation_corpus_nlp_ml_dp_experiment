# config.py
# AIGB 7290 — ESG Litigation Classifier
# Huang, Kiernan, Sooknanan | Fordham University
#
# Single source of truth for all project paths and configuration constants.
# Import in every script: from config import *
#
# Project directory structure:
# Shared drives/ESG DL Project/
# └── esg_project/
#     ├── CLAUDE.md                              # auto-loaded by Claude Code
#     ├── status.md                              # session state — updated iteratively
#     ├── config.py                              # this file
#     ├── 01_esg_deduplicate.py
#     ├── 02_non_esg_filter_noise.py
#     ├── 03_esg_corpus_stats.py
#     ├── 01_manifest.json                       # written by 01_esg_deduplicate.py
#     ├── 02_manifest.json                       # written by 02_non_esg_filter_noise.py
#     ├── 03_manifest.json                       # written by 03_esg_corpus_stats.py
#     ├── esg_corpus/                            # 1,282 raw .md files from zip upload
#     ├── esg_corpus_deduped/                    # 918 unique canonical cases
#     │   └── esg_corpus_dupes/                  # 364 quarantined duplicates (audit only)
#     ├── esg_corpus_filtered/                   # ESG-relevant cases after noise removal
#     │   ├── esg_corpus_excluded/               # quarantined non-ESG noise (audit only)
#     │   ├── esg_filter_log.csv                 # per-file retain/exclude decisions
#     │   ├── esg_corpus_pillar_metadata.csv     # per-file E/S/G signal scores + pillar
#     │   └── esg_corpus_stats.txt               # descriptive statistics report
#     └── esg_corpus_outputs/                    # all generated project artifacts
#         ├── esg_corpus_labels.csv              # label construction output
#         └── [model outputs, reports...]

from pathlib import Path

# ---------------------------------------------------------------------------
# Versioning — applied to all snapshot files, zips, and frozen datasets
# Format: _v{VERSION}_{MDDYYYY}
# Increment VERSION when pipeline decisions change (new labels, re-cleaning, etc.)
# ---------------------------------------------------------------------------
PACKAGE_VERSION = "v1"
PACKAGE_DATE    = "4232026"   # April 23, 2026 — MDDYYYY, matches user zip naming
VERSION_SUFFIX  = f"_{PACKAGE_VERSION}_{PACKAGE_DATE}"

# ---------------------------------------------------------------------------
# Project Root
# Windows path  : C:\GoogleDriveProfiles\Michael_Fordham\Shared drives\ESG DL Project\esg_project
# Google Drive  : /content/drive/Shared Drives/ESG DL Project/esg_project
# Resolves to the directory containing this file regardless of where
# the script is invoked from — safe for Claude Code agentic execution.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Corpus Paths
# ---------------------------------------------------------------------------
ESG_CORPUS          = ROOT / "esg_corpus"               # 1,282 raw .md files from zip
ESG_CORPUS_DEDUPED  = ROOT / "esg_corpus_deduped"       # 918 unique cases
ESG_CORPUS_DUPES    = ESG_CORPUS_DEDUPED / "esg_corpus_dupes"   # quarantined duplicates (audit only)
ESG_CORPUS_FILTERED = ROOT / "esg_corpus_filtered"      # ESG-relevant cases only
ESG_CORPUS_EXCLUDED = ESG_CORPUS_FILTERED / "esg_corpus_excluded"  # quarantined noise (audit only)

# ---------------------------------------------------------------------------
# Output Paths
# ---------------------------------------------------------------------------
ESG_CORPUS_OUTPUTS             = ROOT / "esg_corpus_outputs"
ESG_FILTER_LOG                 = ESG_CORPUS_FILTERED / "esg_filter_log.csv"
ESG_CORPUS_PILLAR_METADATA_CSV = ESG_CORPUS_FILTERED / "esg_corpus_pillar_metadata.csv"
ESG_CORPUS_STATS_REPORT        = ESG_CORPUS_FILTERED / "esg_corpus_stats.txt"
ESG_CORPUS_LABELS_CSV          = ESG_CORPUS_OUTPUTS  / "esg_corpus_labels.csv"
ESG_CLEANED_CSV                = ESG_CORPUS_OUTPUTS  / "ESG_corpus_cleaned_v1.csv"
ESG_ML_BASELINE_DIR            = ESG_CORPUS_OUTPUTS  / "ml_baseline"
ESG_LONGFORMER_DIR             = ESG_CORPUS_OUTPUTS  / "longformer"
ESG_VISUALIZATIONS_DIR         = ESG_CORPUS_OUTPUTS  / "visualizations"

# ---------------------------------------------------------------------------
# Manifest Paths
# Written by each preprocessing script on successful completion.
# Scripts read these to determine whether a step can be skipped.
# Short names (01/02/03_manifest.json) used throughout — no long-form variants.
# ---------------------------------------------------------------------------
MANIFEST_STEP1 = ROOT / "01_manifest.json"     # 01_esg_deduplicate.py
MANIFEST_STEP2 = ROOT / "02_manifest.json"     # 02_non_esg_filter_noise.py
MANIFEST_STEP3 = ROOT / "03_manifest.json"     # 03_esg_corpus_stats.py
MANIFEST_STEP4 = ROOT / "04_manifest.json"     # 04_esg_label_construction.py
MANIFEST_STEP5 = ROOT / "05_manifest.json"     # 05_esg_text_clean.py
MANIFEST_STEP6 = ROOT / "06_manifest.json"     # 06_esg_ml_baseline.py
MANIFEST_STEP7 = ROOT / "07_manifest.json"     # 07_esg_longformer.py
MANIFEST_STEP8 = ROOT / "08_manifest.json"     # 08_esg_xai_visualizations.py

# ---------------------------------------------------------------------------
# Classification Constants
# ---------------------------------------------------------------------------

# Minimum ESG signal hits required to retain a file despite noise domain hits
ESG_OVERRIDE_THRESHOLD = 3

# Expected corpus counts — used for sanity checks in scripts
ESG_EXPECTED_RAW_FILES    = 1282
ESG_EXPECTED_UNIQUE_CASES = 918
ESG_EXPECTED_DUPLICATES   = 364
ESG_KNOWN_GREENWASH_COUNT = 3   # severe class imbalance — flag in methodology

# ---------------------------------------------------------------------------
# Directory Initialization
# Creates all required directories on import if they do not already exist.
# Safe to call repeatedly — mkdir with exist_ok=True is idempotent.
# ---------------------------------------------------------------------------
_required_dirs = [
    ESG_CORPUS,
    ESG_CORPUS_DEDUPED,
    ESG_CORPUS_DUPES,
    ESG_CORPUS_FILTERED,
    ESG_CORPUS_EXCLUDED,
    ESG_CORPUS_OUTPUTS,
]

for _dir in _required_dirs:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Sanity Check (optional — run directly to verify paths)
# python config.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("ESG Project Configuration")
    print("=" * 50)
    print(f"Project root               : {ROOT}")
    print()
    print("Corpus paths:")
    print(f"  Raw corpus               : {ESG_CORPUS}")
    print(f"  Deduped                  : {ESG_CORPUS_DEDUPED}")
    print(f"  Duplicates (audit)       : {ESG_CORPUS_DUPES}")
    print(f"  Filtered                 : {ESG_CORPUS_FILTERED}")
    print(f"  Excluded (audit)         : {ESG_CORPUS_EXCLUDED}")
    print()
    print("Output paths:")
    print(f"  Outputs root             : {ESG_CORPUS_OUTPUTS}")
    print(f"  Filter log               : {ESG_FILTER_LOG}")
    print(f"  Pillar metadata CSV      : {ESG_CORPUS_PILLAR_METADATA_CSV}")
    print(f"  Stats report             : {ESG_CORPUS_STATS_REPORT}")
    print(f"  Labels CSV               : {ESG_CORPUS_LABELS_CSV}")
    print()
    print("Manifest paths:")
    print(f"  Step 1                   : {MANIFEST_STEP1}")
    print(f"  Step 2                   : {MANIFEST_STEP2}")
    print(f"  Step 3                   : {MANIFEST_STEP3}")
    print()
    print("Constants:")
    print(f"  ESG override threshold   : {ESG_OVERRIDE_THRESHOLD}")
    print(f"  Expected raw files       : {ESG_EXPECTED_RAW_FILES}")
    print(f"  Expected unique cases    : {ESG_EXPECTED_UNIQUE_CASES}")
    print(f"  Expected duplicates      : {ESG_EXPECTED_DUPLICATES}")
    print(f"  Known greenwash cases    : {ESG_KNOWN_GREENWASH_COUNT}  *** severe imbalance ***")
    print()
    print("Directory status:")
    for d in _required_dirs:
        status = "exists" if d.exists() else "MISSING"
        print(f"  {status:<8} {d}")
