"""
01_deduplicate.py
ESG Corpus Preprocessing — Step 1: Deduplication
AIGB 7290 | Huang, Kiernan, Sooknanan

Removes duplicate .md files, retaining one canonical version per unique
case name. Duplicates are identified by stripping the numeric prefix
(e.g., "001-", "02-") from each filename. The retained canonical file
is the one with the lowest numeric prefix. All others are moved to a
/duplicates subfolder for audit purposes — nothing is permanently deleted.

Writes 01_manifest.json on completion. If the manifest exists and reports
status "complete", the script skips execution — safe for agentic reruns.
Use --force to rerun regardless of manifest state.

Usage:
    python 01_deduplicate.py --input_dir ./corpus --output_dir ./corpus_deduped
    python 01_deduplicate.py --force   # rerun even if manifest says complete
"""

import os
import re
import json
import shutil
import argparse
from datetime import datetime, timezone
from collections import defaultdict
from config import (
    ESG_CORPUS, ESG_CORPUS_DEDUPED, ESG_CORPUS_DUPES,
    MANIFEST_STEP1
)

MANIFEST_PATH = str(MANIFEST_STEP1)  # absolute, guaranteed correct


def load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {}


def write_manifest(data: dict) -> None:
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(MANIFEST_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Manifest written : {MANIFEST_PATH}")


def extract_case_name(filename: str) -> str:
    """Strip numeric prefix and .md extension to get normalized case name."""
    name = os.path.splitext(filename)[0]
    name = re.sub(r'^[0-9]+-', '', name)
    return name


def deduplicate_corpus(input_dir: str, output_dir: str) -> tuple:
    duplicates_dir = str(ESG_CORPUS_DUPES)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(duplicates_dir, exist_ok=True)

    case_groups = defaultdict(list)
    for fname in os.listdir(input_dir):
        if fname.endswith(".md"):
            case_groups[extract_case_name(fname)].append(fname)

    canonical_count = 0
    duplicate_count = 0

    for case_name, files in case_groups.items():
        files_sorted = sorted(
            files,
            key=lambda f: int(re.match(r'^([0-9]+)', f).group(1))
        )
        canonical  = files_sorted[0]
        duplicates = files_sorted[1:]

        shutil.copy2(os.path.join(input_dir, canonical),
                     os.path.join(output_dir, canonical))
        canonical_count += 1

        for dup in duplicates:
            shutil.copy2(os.path.join(input_dir, dup),
                         os.path.join(duplicates_dir, dup))
            duplicate_count += 1

    print(f"Deduplication complete.")
    print(f"  Canonical files retained : {canonical_count}")
    print(f"  Duplicates quarantined   : {duplicate_count}")
    print(f"  Total input files        : {canonical_count + duplicate_count}")
    print(f"  Output dir               : {output_dir}")
    print(f"  Duplicates dir           : {duplicates_dir}")

    return canonical_count, duplicate_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate ESG corpus .md files.")
    parser.add_argument("--input_dir",  default=str(ESG_CORPUS),         help="Path to raw corpus directory")
    parser.add_argument("--output_dir", default=str(ESG_CORPUS_DEDUPED), help="Path for deduplicated output")
    parser.add_argument("--force",      action="store_true",         help="Rerun even if manifest reports complete")
    args = parser.parse_args()

    manifest = load_manifest()
    if manifest.get("status") == "complete" and not args.force:
        print(f"Step 1 already complete per {MANIFEST_PATH}. Use --force to rerun.")
        print(f"  Canonical  : {manifest.get('canonical_count')}")
        print(f"  Duplicates : {manifest.get('duplicate_count')}")
    else:
        canonical_count, duplicate_count = deduplicate_corpus(args.input_dir, args.output_dir)
        write_manifest({
            "step":            "01_deduplicate",
            "status":          "complete",
            "input_dir":       args.input_dir,
            "output_dir":      args.output_dir,
            "canonical_count": canonical_count,
            "duplicate_count": duplicate_count,
            "total_input":     canonical_count + duplicate_count,
        })
