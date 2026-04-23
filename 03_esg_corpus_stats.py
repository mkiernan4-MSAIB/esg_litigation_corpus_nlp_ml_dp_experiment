"""
03_corpus_stats.py
ESG Corpus Preprocessing — Step 3: Descriptive Statistics
AIGB 7290 | Huang, Kiernan, Sooknanan

Generates a descriptive profile of the filtered corpus covering:
  - File count and size distribution
  - ESG pillar signal frequency (E, S, G, Sustainability)
  - Legal domain distribution
  - Year distribution (extracted from citation strings)
  - Jurisdiction/court distribution
  - Greenwashing and class imbalance flags

Outputs:
  corpus_metadata.csv  — per-file metadata for label construction phase
  corpus_stats.txt     — human-readable descriptive statistics report
  03_manifest.json     — pipeline state for agentic reruns

Checks 03_manifest.json before running. Use --force to rerun.

Usage:
    python 03_corpus_stats.py --input_dir ./corpus_filtered
    python 03_corpus_stats.py --force
"""

import os
import re
import csv
import json
import argparse
from datetime import datetime, timezone
from collections import Counter

from config import (
    ESG_CORPUS_FILTERED,
    ESG_CORPUS_PILLAR_METADATA_CSV, ESG_CORPUS_STATS_REPORT,
    MANIFEST_STEP3
)

MANIFEST_PATH = str(MANIFEST_STEP3)  # absolute, guaranteed correct

# Pillar signal keyword sets — corpus-agnostic, from agreed legal definition
PILLAR_E = [
    r'\bclimate\b', r'\bgreenhouse\b', r'\bscope [123]\b', r'\bcarbon\b',
    r'\bpollution\b', r'\bclean air act\b', r'\bclean water act\b',
    r'\bcercla\b', r'\bepa\b', r'\bgreenwash', r'\bnet zero\b',
    r'\bbiodiversity\b', r'\bemissions\b', r'\bcsrd\b', r'\bsb 253\b',
]

PILLAR_S = [
    r'\bhuman rights\b', r'\bsupply chain\b', r'\bforced labor\b',
    r'\buflpa\b', r'\bdei\b', r'\bdiversity\b', r'\bworker safety\b',
    r'\bosha\b', r'\blabor rights\b', r'\bhuman capital\b',
    r'\btitle vii\b', r'\bflsa\b', r'\bnlra\b', r'\bcsddd\b',
    r'\bconsumer protection\b', r'\bpolitical spending\b',
]

PILLAR_G = [
    r'\bfiduciary\b', r'\berisa\b', r'\bproxy voting\b',
    r'\bboard composition\b', r'\bexecutive compensation\b',
    r'\bshareholder rights\b', r'\bsecurities fraud\b',
    r'\bdisclosure\b', r'\bfcpa\b', r'\bsarbanes\b', r'\bdodd.frank\b',
    r'\bcorporate governance\b', r'\bdol\b', r'\b29 cfr\b',
]

PILLAR_SUS = [
    r'\bsustainab', r'\bbrundtland\b', r'\blong.term viability\b',
    r'\bnet zero\b', r'\bcircular economy\b', r'\bsdg\b',
    r'\bresponsible investment\b', r'\bpri\b',
]

NOISE_DOMAINS = [
    r'\bcopyright\b', r'\btrademark\b', r'\bpatent\b',
    r'\bantitrust\b', r'\bsherman act\b',
]

YEAR_PATTERN = re.compile(r'\b(20[0-9]{2})\b')

COURT_PATTERNS = {
    "SDNY":    r'S\.D\. New York|Southern District of New York',
    "CD Cal":  r'C\.D\. California|Central District of California',
    "ND Cal":  r'N\.D\. California|Northern District of California',
    "ND Tex":  r'N\.D\. Texas',
    "SD Tex":  r'S\.D\. Texas',
    "9th Cir": r'Ninth Circuit',
    "2nd Cir": r'Second Circuit',
    "5th Cir": r'Fifth Circuit',
    "DC Cir":  r'District of Columbia Circuit|D\.C\. Circuit',
    "Del Ch":  r'Delaware Court of Chancery|Del\. Ch\.',
    "SCOTUS":  r'Supreme Court of the United States',
}


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


def extract_year(text: str) -> str:
    years = YEAR_PATTERN.findall(text[:2000])
    return Counter(years).most_common(1)[0][0] if years else "unknown"


def detect_court(text: str) -> str:
    for label, pattern in COURT_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            return label
    return "Other"


def infer_primary_pillar(e: int, s: int, g: int) -> str:
    scores = {"E": e, "S": s, "G": g}
    top = max(scores, key=scores.get)
    return top if scores[top] > 0 else "Non-ESG"


def analyze_corpus(input_dir: str) -> dict:
    files = [f for f in sorted(os.listdir(input_dir)) if f.endswith(".md")]

    pillar_counts = Counter()
    year_counts   = Counter()
    court_counts  = Counter()
    metadata_rows = []
    total_bytes   = 0
    greenwash_count = 0

    for fname in files:
        fpath = os.path.join(input_dir, fname)
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        size = os.path.getsize(fpath)
        total_bytes += size

        e   = count_matches(text, PILLAR_E)
        s   = count_matches(text, PILLAR_S)
        g   = count_matches(text, PILLAR_G)
        sus = count_matches(text, PILLAR_SUS)
        noise  = count_matches(text, NOISE_DOMAINS)
        year   = extract_year(text)
        court  = detect_court(text)
        pillar = infer_primary_pillar(e, s, g)
        is_sustainability = 1 if sus > 0 else 0
        is_greenwash = 1 if re.search(r'greenwash', text, re.IGNORECASE) else 0

        greenwash_count += is_greenwash
        pillar_counts[pillar] += 1
        year_counts[year]     += 1
        court_counts[court]   += 1

        metadata_rows.append({
            "filename":          fname,
            "size_bytes":        size,
            "year":              year,
            "court":             court,
            "pillar_E":          e,
            "pillar_S":          s,
            "pillar_G":          g,
            "pillar_Sus":        sus,
            "noise_hits":        noise,
            "is_greenwash":      is_greenwash,
            "is_sustainability": is_sustainability,
            "inferred_pillar":   pillar,
        })

    # Write metadata CSV
    csv_path = str(ESG_CORPUS_PILLAR_METADATA_CSV)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metadata_rows[0].keys())
        writer.writeheader()
        writer.writerows(metadata_rows)

    # Write stats report
    report_path = str(ESG_CORPUS_STATS_REPORT)
    n = len(files)
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("ESG CORPUS DESCRIPTIVE STATISTICS\n")
        f.write("AIGB 7290 | Huang, Kiernan, Sooknanan\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total files analyzed  : {n}\n")
        f.write(f"Total corpus size     : {total_bytes / 1e6:.1f} MB\n")
        f.write(f"Avg file size         : {total_bytes / n / 1e3:.1f} KB\n\n")

        f.write("--- Inferred Pillar Distribution ---\n")
        for pillar, count in pillar_counts.most_common():
            f.write(f"  {pillar:<20} {count:>5}  ({count/n*100:.1f}%)\n")

        f.write("\n--- Year Distribution (Top 10) ---\n")
        for year, count in year_counts.most_common(10):
            f.write(f"  {year}  {count}\n")

        f.write("\n--- Court/Jurisdiction Distribution ---\n")
        for court, count in court_counts.most_common():
            f.write(f"  {court:<20} {count}\n")

        f.write("\n--- Class Imbalance Flags ---\n")
        f.write(f"  Greenwashing cases   : {greenwash_count}")
        if greenwash_count < 10:
            f.write("  *** SEVERE IMBALANCE — address via oversampling or synthetic augmentation ***")
        f.write("\n")

    print(f"Analysis complete.")
    print(f"  Files analyzed   : {n}")
    print(f"  Metadata CSV     : {csv_path}")
    print(f"  Stats report     : {report_path}")
    print(f"  Greenwash cases  : {greenwash_count}")
    print(f"\nPillar distribution:")
    for pillar, count in pillar_counts.most_common():
        print(f"  {pillar:<20} {count} ({count/n*100:.1f}%)")

    return {
        "files_analyzed":   n,
        "greenwash_count":  greenwash_count,
        "pillar_counts":    dict(pillar_counts),
        "csv_path":         csv_path,
        "report_path":      report_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate descriptive statistics for filtered ESG corpus.")
    parser.add_argument("--input_dir", default=str(ESG_CORPUS_FILTERED), help="Path to filtered corpus")
    parser.add_argument("--force",     action="store_true",          help="Rerun even if manifest reports complete")
    args = parser.parse_args()

    manifest = load_manifest()
    if manifest.get("status") == "complete" and not args.force:
        print(f"Step 3 already complete per {MANIFEST_PATH}. Use --force to rerun.")
        print(f"  Files analyzed  : {manifest.get('files_analyzed')}")
        print(f"  Greenwash cases : {manifest.get('greenwash_count')}")
    else:
        results = analyze_corpus(args.input_dir)
        write_manifest({
            "step":           "03_corpus_stats",
            "status":         "complete",
            "input_dir":      args.input_dir,
            "files_analyzed": results["files_analyzed"],
            "greenwash_count": results["greenwash_count"],
            "pillar_counts":  results["pillar_counts"],
            "csv_path":       results["csv_path"],
            "report_path":    results["report_path"],
        })
