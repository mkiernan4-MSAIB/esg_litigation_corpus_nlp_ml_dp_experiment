"""
05_esg_text_clean.py
ESG Corpus Preprocessing — Step 5: Text Cleaning, Leakage Masking, Token Audit
AIGB 7290 | Huang, Kiernan, Sooknanan

Reads the 444 filtered .md files from esg_corpus_filtered/, applies:
  1. Westlaw header/caption removal
       - User/date stamp, "For Educational Use Only", Thomson Reuters footer
       - Case caption (parties, case number, date, attorneys section)
       - Reporter citations (e.g. "432 F.3d 201", "2017 WL 11679600")
  2. Outcome leakage masking
       Replace outcome-indicating terms with [OUTCOME] to prevent the model
       from inferring labels from procedural resolution rather than case substance.
       Masked in context (preceded by motion/judgment/court language) and as
       standalone ALL-CAPS judicial pronouncements.
  3. Token audit
       Estimate token count via whitespace split (approximate; consistent with
       BERT tokenizer upper bound). Reports % of docs exceeding 512 tokens.

Outputs:
  esg_corpus_outputs/ESG_corpus_cleaned_v1.csv
    columns: filename, label, is_sustainability, is_greenwash,
             cleaned_text, token_count, exceeds_512

Checks 05_manifest.json before running. Use --force to rerun.
"""

import os
import re
import csv
import json
import argparse
from datetime import datetime, timezone
from config import (
    ESG_CORPUS_FILTERED,
    ESG_CORPUS_LABELS_CSV,
    ESG_CORPUS_OUTPUTS,
)

MANIFEST_PATH = str(ESG_CORPUS_OUTPUTS.parent / "05_manifest.json")
OUTPUT_CSV    = str(ESG_CORPUS_OUTPUTS / "ESG_corpus_cleaned_v1.csv")

# ---------------------------------------------------------------------------
# Westlaw structural patterns to strip
# ---------------------------------------------------------------------------
HEADER_PATTERNS = [
    # User/date stamp: "Raghupathi, Wullianallur 4/13/2026"
    re.compile(r'^.{3,60}\d{1,2}/\d{1,2}/\d{4}\s*$', re.MULTILINE),
    # For Educational Use Only
    re.compile(r'For Educational Use Only', re.IGNORECASE),
    # Thomson Reuters copyright
    re.compile(r'©?\s*\d{4}\s*Thomson Reuters\..*?Works\.', re.IGNORECASE | re.DOTALL),
    # WL citation lines: "2017 WL 11679600"
    re.compile(r'^\d{4} WL \d+\s*$', re.MULTILINE),
    # "Only the Westlaw citation is currently available."
    re.compile(r'Only the Westlaw citation is currently available\.', re.IGNORECASE),
    # Reporter citations: 432 F.3d 201 / 112 S.Ct. 1029 / 2011 U.S. Dist. LEXIS 999
    re.compile(r'\d+\s+(?:F\.\d[a-z]*|S\.Ct\.|U\.S\.|F\.Supp\.\d*|L\.Ed\.\d*|U\.S\.\s*Dist\.\s*LEXIS)\s+\d+'),
    # Case number lines: "Case No. 2:11-cv-2288-SLD-JEH"
    re.compile(r'Case No\.?\s+[\w:\-]+', re.IGNORECASE),
    # Signed/dated lines
    re.compile(r'^\|\s*$', re.MULTILINE),
    re.compile(r'^Signed\s+\d{2}/\d{2}/\d{4}\s*$', re.MULTILINE),
    # Attorneys and Law Firms section (strip through double newline)
    re.compile(r'Attorneys and Law Firms.*?(?=\n[A-Z]{2,}|\Z)', re.DOTALL | re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Outcome leakage — mask with [OUTCOME]
# Context-aware: captures judicial outcome pronouncements.
# Standalone ALL-CAPS forms catch header-style rulings (e.g. "GRANTED").
# ---------------------------------------------------------------------------
LEAKAGE_TERMS = [
    "dismissed", "granted", "affirmed", "reversed",
    "remanded", "vacated", "denied", "overruled",
    "enjoined", "prevailed", "affirming", "reversing",
    "remanding", "vacating",
]

# Contextual pattern: preceded by motion/judgment/court trigger words
_ctx_trigger = r'(?:motion|judgment|appeal|order|verdict|court|hereby|is|are|be|was|were|has been|have been)\s+'
_terms_re    = '|'.join(LEAKAGE_TERMS)
LEAKAGE_CONTEXTUAL = re.compile(
    rf'({_ctx_trigger})({_terms_re})',
    re.IGNORECASE
)
# Standalone ALL-CAPS judicial pronouncements (e.g. "GRANTED", "DENIED")
LEAKAGE_ALLCAPS = re.compile(
    r'\b(' + '|'.join(t.upper() for t in LEAKAGE_TERMS) + r')\b'
)


def strip_headers(text: str) -> str:
    for pattern in HEADER_PATTERNS:
        text = pattern.sub(' ', text)
    # Collapse excess whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def mask_leakage(text: str) -> str:
    text = LEAKAGE_CONTEXTUAL.sub(lambda m: m.group(1) + '[OUTCOME]', text)
    text = LEAKAGE_ALLCAPS.sub('[OUTCOME]', text)
    return text


def token_count(text: str) -> int:
    return len(text.split())


def load_labels(labels_csv: str) -> dict:
    labels = {}
    with open(labels_csv, newline="") as f:
        for row in csv.DictReader(f):
            labels[row["filename"]] = row
    return labels


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


def run_pipeline(input_dir: str, labels_csv: str, output_csv: str) -> dict:
    labels = load_labels(labels_csv)
    files  = sorted(f for f in os.listdir(input_dir) if f.endswith(".md"))

    rows         = []
    over_512     = 0
    total_tokens = 0

    for fname in files:
        fpath = os.path.join(input_dir, fname)
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        cleaned = strip_headers(raw)
        cleaned = mask_leakage(cleaned)
        tokens  = token_count(cleaned)
        total_tokens += tokens
        exceeds = 1 if tokens > 512 else 0
        over_512 += exceeds

        meta = labels.get(fname, {})
        rows.append({
            "filename":          fname,
            "label":             meta.get("label", ""),
            "is_sustainability": meta.get("is_sustainability", ""),
            "is_greenwash":      meta.get("is_greenwash", ""),
            "token_count":       tokens,
            "exceeds_512":       exceeds,
            "cleaned_text":      cleaned,
        })

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    pct_over = over_512 / n * 100
    avg_tok  = total_tokens / n

    print(f"Text cleaning complete.")
    print(f"  Files processed  : {n}")
    print(f"  Output           : {output_csv}")
    print(f"\nToken audit:")
    print(f"  Avg token count  : {avg_tok:.0f}")
    print(f"  Exceeds 512 tok  : {over_512} / {n}  ({pct_over:.1f}%)")
    if pct_over > 20:
        print(f"  *** >20% exceed 512 — Longformer or chunking strategy required ***")
    else:
        print(f"  DistilBERT/RoBERTa 512-token limit is workable for this corpus.")

    return {
        "files_processed": n,
        "over_512_count":  over_512,
        "over_512_pct":    round(pct_over, 2),
        "avg_token_count": round(avg_tok, 1),
        "output_csv":      output_csv,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean ESG corpus text and audit token lengths.")
    parser.add_argument("--input_dir",  default=str(ESG_CORPUS_FILTERED),  help="Filtered corpus directory")
    parser.add_argument("--labels_csv", default=str(ESG_CORPUS_LABELS_CSV), help="Label CSV from Step 4")
    parser.add_argument("--output_csv", default=OUTPUT_CSV,                  help="Output cleaned CSV path")
    parser.add_argument("--force",      action="store_true",                  help="Rerun even if manifest complete")
    args = parser.parse_args()

    manifest = load_manifest()
    if manifest.get("status") == "complete" and not args.force:
        print(f"Step 5 already complete per {MANIFEST_PATH}. Use --force to rerun.")
        print(f"  Files processed : {manifest.get('files_processed')}")
        print(f"  Exceeds 512     : {manifest.get('over_512_count')} ({manifest.get('over_512_pct')}%)")
        print(f"  Avg tokens      : {manifest.get('avg_token_count')}")
    else:
        results = run_pipeline(args.input_dir, args.labels_csv, args.output_csv)
        write_manifest({
            "step":             "05_text_clean",
            "status":           "complete",
            "input_dir":        args.input_dir,
            "labels_csv":       args.labels_csv,
            **results,
        })
