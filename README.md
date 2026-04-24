# ESG Litigation Classifier
## AIGB 7290 Deep Learning | Fordham University
### Huang, Kiernan, Sooknanan

Classifies 1,282 Westlaw litigation documents into Environmental (E), Social (S), Governance (G), and Non-ESG categories using NLP and deep learning. Taxonomy anchored in external regulatory authority only (GHG Protocol, UN Guiding Principles, ERISA § 404, UN Brundtland Commission).

**GitHub:** https://github.com/mkiernan4-MSAIB/esg_litigation_corpus_nlp_ml_dp_experiment

---

## Pipeline

```
esg_corpus/                          1,282 raw Westlaw .md files
    |
    01_esg_deduplicate.py            Cosine similarity dedup (>0.95 threshold)
    |                                -> 01_manifest.json
esg_corpus_deduped/                  918 unique canonical cases
    |
    02_non_esg_filter_noise.py       ESG keyword signal filter
    |                                -> 02_manifest.json
esg_corpus_filtered/                 444 high-signal ESG cases
    |
    03_esg_corpus_stats.py           Descriptive stats + pillar metadata
    |                                -> 03_manifest.json
    |                                -> esg_corpus_pillar_metadata.csv
    |
    04_esg_label_construction.py     argmax(E,S,G); tiebreaker G>E>S
    |                                -> 04_manifest.json
    |                                -> esg_corpus_labels.csv  [AUTHORITATIVE LABELS]
    |
    05_esg_text_clean.py             [OUTCOME] masking + token audit
    |                                -> 05_manifest.json
    |                                -> ESG_corpus_cleaned_v1.csv  [AUTHORITATIVE MODEL INPUT]
    |
    06_esg_ml_baseline.py            TF-IDF + 6 models + SHAP (27 plots)  [COMPLETE]
    |  (CPU-viable)                  -> 06_manifest.json
    |                                -> esg_corpus_outputs/ml_baseline/
    |
    10_esg_descriptive_analysis.py   Word clouds + n-gram extraction (11 PNGs)  [COMPLETE]
    |  (CPU-viable)                  -> 10_manifest.json
    |                                -> esg_corpus_outputs/descriptive_analysis/
    |
    07_esg_longformer.py             Longformer-base-4096 fine-tuning  [IN PROGRESS -- Colab T4]
    |  (Colab GPU required)          3 seeds x 3 folds x 5 epochs, AMP FP16
    |  AMP + grad checkpointing      Resume via 07_progress.json (fold-level)
    |  Browser keepalive built-in    -> 07_manifest.json
    |                                -> esg_corpus_outputs/longformer/
    |
    08_esg_xai_visualizations.py     ROC, word clouds, SHAP, t-SNE, attention  [PENDING -- blocked on 07]
    |  (run after 07)                -> 08_manifest.json
    |                                -> esg_corpus_outputs/visualizations/
    |
    09_create_reproducibility_package.py
                                     Versioned snapshots + per-step zips + SHA-256  [COMPLETE]
                                     -> 09_manifest.json
                                     -> snapshots/  step_zips/
    |
    06_create_reproducibility_zip.py  Step-06 zip regenerated for revised 27-plot outputs  [COMPLETE]
    10_create_reproducibility_zip.py  Step-10 zip  [COMPLETE]
```

---

## Corpus Statistics

| Stage | Count |
|---|---|
| Raw Westlaw documents | 1,282 |
| After deduplication | 918 |
| After noise filtering | 444 |
| **Model input (filtered + cleaned)** | **444** |

**Label distribution:** G 171 (38.5%) | Non-ESG 134 (30.2%) | E 92 (20.7%) | S 47 (10.6%)  
**is_sustainability flag:** 64 cases | **is_greenwash:** 1 case  
**Token audit:** 99.1% exceed 512 tokens | Mean 8,824 tokens/case → Longformer required

---

## ML Baseline Results (April 23, 2026 — revised)

Test set n=67 | Stratified 70/15/15 split | TF-IDF ngram(1,2) min_df=3

| Model | Macro-F1 | MCC | AUC (OvR) |
|---|---|---|---|
| Majority Baseline | 0.1398 | 0.0000 | 0.5000 |
| BoW Logistic Regression | 0.7529 | 0.6635 | 0.9135 |
| Logistic Regression | 0.7409 | 0.6666 | 0.9056 |
| Complement Naive Bayes | 0.6117 | 0.5806 | 0.8759 |
| Random Forest | 0.6760 | 0.6860 | 0.9287 |
| **XGBoost** | **0.7684** | **0.7467** | **0.9663** |

XGBoost is the authoritative ML baseline. 27 plots produced (confusion matrices, per-pillar ROC curves, SHAP beeswarms/waterfall/bar). Longformer results pending Colab GPU execution.

---

## Reproducibility

Every pipeline step produces a versioned snapshot (`_v1_4232026`) with SHA-256 integrity hashes.

**Frozen training input anchor:**
```
feature_matrix_v1_frozen.csv
SHA-256: a2b95dfd616fc8573c1f27a4d4a4b18c02136c011a9ded6772b28bb75a00283e
```

Per-step reproducibility zips in `step_zips/`:

| Zip | Contents | SHA-256 (first 16) |
|---|---|---|
| `snapshot_01_deduplication_v1_4232026.zip` | CSV, JSON, manifest, script | `ae7f5a54...` |
| `snapshot_02_noise_filtering_v1_4232026.zip` | + filter_log.csv | `d67fbb4e...` |
| `snapshot_03_corpus_stats_v1_4232026.zip` | + pillar metadata, stats.txt | `75cc7417...` |
| `snapshot_04_label_construction_v1_4232026.zip` | + esg_corpus_labels.csv | `97ee91e3...` |
| `snapshot_05_text_cleaning_v1_4232026.zip` | + feature_matrix_v1_frozen.csv | `aaa9f821...` |
| `snapshot_06_ml_baseline_v1_4232026.zip` | 24 PNGs, 5 pkl models, metrics, predictions | `b7b3e40e...` |
| `snapshot_10_descriptive_analysis_v1_4232026.zip` | 11 PNGs, n-gram CSV | `eb39099f...` |

Each zip contains `reproducibility_manifest_v1_4232026.json` with SHA-256 for every file.

To verify integrity:
```python
import hashlib
h = hashlib.sha256()
with open("esg_corpus_outputs/feature_matrix_v1_frozen.csv", "rb") as f:
    for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
assert h.hexdigest() == "a2b95dfd616fc8573c1f27a4d4a4b18c02136c011a9ded6772b28bb75a00283e"
```

---

## Key Output Files

| File | Description |
|---|---|
| `esg_corpus_outputs/esg_corpus_labels.csv` | Authoritative label source |
| `esg_corpus_outputs/ESG_corpus_cleaned_v1.csv` | Authoritative model input (24MB, git-excluded) |
| `esg_corpus_outputs/feature_matrix_v1_frozen.csv` | Frozen training input with SHA-256 anchor |
| `esg_corpus_outputs/ml_baseline/` | RF + XGBoost models, SHAP plots, metrics JSON |
| `esg_corpus_filtered/esg_corpus_pillar_metadata.csv` | Per-file E/S/G signal scores |
| `snapshots/` | 6 versioned snapshot pairs (CSV + JSON) |
| `step_zips/` | 6 per-step reproducibility zips |
| `esg_slide_narrative.md` | Methodology + Limitations slide prose (16 citations) |

---

## Version Control

- **SSH auth:** ed25519 key configured. Push with `git push origin master`.
- **Versioning scheme:** `_{VERSION}_{MDDYYYY}` suffix on all snapshots and zips.
- Increment `PACKAGE_VERSION` in `config.py` when pipeline decisions change.
- Raw corpus `.md` files and large CSVs (24MB+) are git-excluded per `.gitignore`.

---

## Notes

- Nothing is permanently deleted — all excluded files are quarantined in `esg_corpus_dupes/` and `esg_corpus_excluded/`.
- Sustainability is a cross-cutting binary modifier (`is_sustainability`), not a primary class.
- Class imbalance (S = 10.6%, greenwash = 1 case) addressed via class-weighted loss; address in limitations slide.
- 2 greenwash cases lost as false negatives in noise filter — address in limitations slide.
- All keyword patterns derived from corpus-agnostic external regulatory authority only.
