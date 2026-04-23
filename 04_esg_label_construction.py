"""
04_esg_label_construction.py
ESG Corpus Preprocessing — Step 4: Label Construction
AIGB 7290 | Huang, Kiernan, Sooknanan

Reads esg_corpus_pillar_metadata.csv (produced by Step 3) and applies
deterministic decision rules to assign a primary label (E, S, G, Non-ESG)
to each of the 444 filtered cases.

Decision rules:
  1. max(E, S, G) == 0  → Non-ESG
  2. max(E, S, G) >= 1  → argmax pillar
  3. Tiebreaker (equal top scores): G > E > S
     Rationale: Governance is most legally well-defined (ERISA, SEC, SOX);
     Environmental carries more specific regulatory anchors than Social;
     Social claims most frequently co-occur with G or E.
  4. is_sustainability: passthrough binary flag (sus_score > 0)
  5. is_greenwash: passthrough binary flag from metadata

Output columns:
  filename, label, is_sustainability, is_greenwash,
  E_score, S_score, G_score, sus_score, signal_strength

Writes 04_manifest.json on completion. Use --force to rerun.
"""

import csv
import json
import argparse
from datetime import datetime, timezone
from collections import Counter
from config import (
    ESG_CORPUS_PILLAR_METADATA_CSV,
    ESG_CORPUS_LABELS_CSV,
)

MANIFEST_PATH = str(
    ESG_CORPUS_PILLAR_METADATA_CSV.parent.parent / "04_manifest.json"
)

TIEBREAK_ORDER = ["G", "E", "S"]


def load_manifest() -> dict:
    try:
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def write_manifest(data: dict) -> None:
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(MANIFEST_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Manifest written : {MANIFEST_PATH}")


def assign_label(e: int, s: int, g: int) -> str:
    if max(e, s, g) == 0:
        return "Non-ESG"
    scores = {"E": e, "S": s, "G": g}
    top_score = max(scores.values())
    candidates = [p for p in TIEBREAK_ORDER if scores[p] == top_score]
    return candidates[0]


def build_labels(metadata_path: str, output_path: str) -> dict:
    with open(metadata_path, newline="") as f:
        rows = list(csv.DictReader(f))

    label_rows = []
    for r in rows:
        e   = int(r["pillar_E"])
        s   = int(r["pillar_S"])
        g   = int(r["pillar_G"])
        sus = int(r["pillar_Sus"])

        label = assign_label(e, s, g)
        label_rows.append({
            "filename":          r["filename"],
            "label":             label,
            "is_sustainability": int(r["is_sustainability"]),
            "is_greenwash":      int(r["is_greenwash"]),
            "E_score":           e,
            "S_score":           s,
            "G_score":           g,
            "sus_score":         sus,
            "signal_strength":   max(e, s, g),
        })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=label_rows[0].keys())
        writer.writeheader()
        writer.writerows(label_rows)

    dist = Counter(r["label"] for r in label_rows)
    sus_count = sum(r["is_sustainability"] for r in label_rows)
    gw_count  = sum(r["is_greenwash"] for r in label_rows)
    n = len(label_rows)

    print(f"Label construction complete.")
    print(f"  Total cases      : {n}")
    print(f"  Output           : {output_path}")
    print(f"\nLabel distribution:")
    for label in ["E", "S", "G", "Non-ESG"]:
        count = dist[label]
        print(f"  {label:<12} {count:>4}  ({count/n*100:.1f}%)")
    print(f"\n  is_sustainability : {sus_count} ({sus_count/n*100:.1f}%)")
    print(f"  is_greenwash      : {gw_count}")

    return {
        "total_cases":       n,
        "label_distribution": dict(dist),
        "sustainability_flagged": sus_count,
        "greenwash_count":   gw_count,
        "output_path":       output_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct ESG labels from pillar metadata.")
    parser.add_argument("--metadata", default=str(ESG_CORPUS_PILLAR_METADATA_CSV), help="Path to pillar metadata CSV")
    parser.add_argument("--output",   default=str(ESG_CORPUS_LABELS_CSV),          help="Path for output labels CSV")
    parser.add_argument("--force",    action="store_true",                          help="Rerun even if manifest reports complete")
    args = parser.parse_args()

    manifest = load_manifest()
    if manifest.get("status") == "complete" and not args.force:
        print(f"Step 4 already complete per {MANIFEST_PATH}. Use --force to rerun.")
        dist = manifest.get("label_distribution", {})
        for label in ["E", "S", "G", "Non-ESG"]:
            print(f"  {label}: {dist.get(label, 0)}")
    else:
        results = build_labels(args.metadata, args.output)
        write_manifest({
            "step":                  "04_label_construction",
            "status":                "complete",
            "metadata_path":         args.metadata,
            "output_path":           args.output,
            "total_cases":           results["total_cases"],
            "label_distribution":    results["label_distribution"],
            "sustainability_flagged": results["sustainability_flagged"],
            "greenwash_count":       results["greenwash_count"],
        })
