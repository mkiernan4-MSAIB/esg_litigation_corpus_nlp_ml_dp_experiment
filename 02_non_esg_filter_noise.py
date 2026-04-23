"""
02_filter_noise.py
ESG Corpus Preprocessing — Step 2: Non-ESG Noise Filtering
AIGB 7290 | Huang, Kiernan, Sooknanan

Applies inclusion/exclusion criteria derived from the project's agreed
corpus-agnostic legal definition of ESG + Sustainability to remove cases
with no nexus to environmental conduct, treatment of people, governance
structures, or sustainability disclosures.

Exclusion logic is two-stage:
  Stage 1 — Hard exclusion: files where noise domain keywords dominate
             and ESG signal is below threshold are excluded.
  Stage 2 — ESG signal check: any file passing Stage 1 must contain at
             least one affirmative ESG signal keyword to be retained.

Excluded files are quarantined to /excluded for audit — nothing is deleted.

Checks 02_manifest.json before running. Use --force to rerun.

Usage:
    python 02_filter_noise.py --input_dir ./corpus_deduped --output_dir ./corpus_filtered
    python 02_filter_noise.py --force
"""

import os
import re
import csv
import json
import shutil
import argparse
from datetime import datetime, timezone
from config import (
    ESG_CORPUS_DEDUPED, ESG_CORPUS_FILTERED,
    ESG_CORPUS_EXCLUDED, ESG_FILTER_LOG,
    MANIFEST_STEP2
)

MANIFEST_PATH = str(MANIFEST_STEP2)  # absolute, guaranteed correct



# Hard exclusion: dominant legal domains with no ESG nexus
NOISE_PATTERNS = [
    r'\bcopyright infringement\b',
    r'\bcopyright act\b',
    r'\b17 u\.s\.c\b',
    r'\btrademark infringement\b',
    r'\blanham act\b',
    r'\btrade secret\b',
    r'\bpatent infringement\b',
    r'\bdefamation\b',
    r'\blibel\b',
    r'\bslander\b',
]

# ESG affirmative signals — at least one required to retain
ESG_SIGNALS = [
    r'\besg\b', r'\benvironmental\b', r'\bgreenwash', r'\bsustainab',
    r'\bclimate\b', r'\bgreenhouse\b', r'\bscope [123]\b', r'\bcarbon\b',
    r'\bpollution\b', r'\bclean air act\b', r'\bclean water act\b',
    r'\bcercla\b', r'\bepa\b', r'\bgovernance\b', r'\bfiduciary\b',
    r'\berisa\b', r'\bproxy voting\b', r'\bboard composition\b',
    r'\bexecutive compensation\b', r'\bshareholder rights\b',
    r'\bsecurities fraud\b', r'\bdisclosure\b', r'\bdei\b',
    r'\bdiversity\b', r'\bhuman rights\b', r'\bsupply chain\b',
    r'\bforced labor\b', r'\blabor rights\b', r'\bhuman capital\b',
    r'\bworker safety\b', r'\bosha\b', r'\bnet zero\b',
]

# Number of ESG signal matches required to retain despite noise hits
ESG_OVERRIDE_THRESHOLD = 3


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


def count_matches(text: str, patterns: list) -> int:
    tl = text.lower()
    return sum(1 for p in patterns if re.search(p, tl))


def classify_file(text: str) -> str:
    noise_hits = count_matches(text, NOISE_PATTERNS)
    esg_hits   = count_matches(text, ESG_SIGNALS)
    if noise_hits > 0 and esg_hits < ESG_OVERRIDE_THRESHOLD:
        return "exclude"
    if esg_hits == 0:
        return "exclude"
    return "retain"


def filter_corpus(input_dir: str, output_dir: str) -> tuple:
    excluded_dir = str(ESG_CORPUS_EXCLUDED)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(excluded_dir, exist_ok=True)

    retained  = 0
    excluded  = 0
    log_rows  = []

    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".md"):
            continue
        fpath = os.path.join(input_dir, fname)
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        decision   = classify_file(text)
        noise_hits = count_matches(text, NOISE_PATTERNS)
        esg_hits   = count_matches(text, ESG_SIGNALS)
        log_rows.append({
            "decision":   decision,
            "filename":   fname,
            "noise_hits": noise_hits,
            "esg_hits":   esg_hits,
        })

        if decision == "retain":
            shutil.copy2(fpath, os.path.join(output_dir, fname))
            retained += 1
        else:
            shutil.copy2(fpath, os.path.join(excluded_dir, fname))
            excluded += 1

    # Write audit log
    log_path = str(ESG_FILTER_LOG)
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["decision", "filename", "noise_hits", "esg_hits"])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"Filtering complete.")
    print(f"  Files retained : {retained}")
    print(f"  Files excluded : {excluded}")
    print(f"  Audit log      : {log_path}")

    return retained, excluded, log_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter non-ESG noise from corpus.")
    parser.add_argument("--input_dir",  default=str(ESG_CORPUS_DEDUPED),  help="Path to deduplicated corpus")
    parser.add_argument("--output_dir", default=str(ESG_CORPUS_FILTERED),  help="Path for filtered output")
    parser.add_argument("--force",      action="store_true",           help="Rerun even if manifest reports complete")
    args = parser.parse_args()

    manifest = load_manifest()
    if manifest.get("status") == "complete" and not args.force:
        print(f"Step 2 already complete per {MANIFEST_PATH}. Use --force to rerun.")
        print(f"  Retained : {manifest.get('retained_count')}")
        print(f"  Excluded : {manifest.get('excluded_count')}")
    else:
        retained, excluded, log_path = filter_corpus(args.input_dir, args.output_dir)
        write_manifest({
            "step":           "02_filter_noise",
            "status":         "complete",
            "input_dir":      args.input_dir,
            "output_dir":     args.output_dir,
            "retained_count": retained,
            "excluded_count": excluded,
            "audit_log":      log_path,
        })
