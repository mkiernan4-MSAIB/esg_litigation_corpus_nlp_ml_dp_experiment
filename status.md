# Session Status — AIGB 7290 ESG Deep Learning Project
**Date:** April 23, 2026 | **Team:** James Huang, Michael Kiernan, Amanda Sooknanan

---

## 1. Current Task Status

**Session April 23, 2026 — ML BASELINE REVISED AND RE-EXECUTED. DESCRIPTIVE ANALYSIS EXECUTED. REPRODUCIBILITY PACKAGE COMPLETE. VERSION CONTROL ACTIVE. LONGFORMER TRAINING IN PROGRESS (COLAB T4).**

Preprocessing pipeline complete (scripts 01–05). ML baseline executed locally (script 06) — 27 plots, all bugs resolved, reproducibility zip regenerated (23.6 MB). Descriptive analysis executed locally (script 10) — 11 PNGs (6 word clouds + 4 per-pillar n-gram panels + 1 composite) + n-gram CSV produced. Script 07 revised with AMP, gradient checkpointing, fold-level resume, tqdm progress bars, and browser keepalive — currently executing in Colab T4 (seed 42 in progress). Script 08 pending Longformer completion. Versioned reproducibility package (per-step zips + Jupyter notebook) committed and pushed to GitHub. PowerPoint revision (9 slides) still pending — slide narrative prose ready in `esg_slide_narrative.md`.

**GitHub repo:** https://github.com/mkiernan4-MSAIB/esg_litigation_corpus_nlp_ml_dp_experiment  
**Latest commit:** aa54704 — versioned filenames + SHA-256 integrity hashes  
**Auth method:** SSH (ed25519). Remote set to `git@github.com:mkiernan4-MSAIB/...`. No PATs required. Push with `git push origin master`.

**REMAINING EXECUTION:**
1. Run `07_esg_longformer.py` in Colab T4/A100 — produces Longformer checkpoints + CLS embeddings.
2. Run `08_esg_xai_visualizations.py` after 07 completes — produces ROC, word clouds, SHAP, t-SNE, attention heatmaps.
3. Produce PowerPoint deck in Claude.ai using `esg_slide_narrative.md` as content source.

---

## 2. Decisions Made

**Corpus:** 1,282 .md files. 364 duplicates removed (918 unique). 474 excluded as non-ESG noise. 444 cases in filtered corpus.

**ESG + Sustainability Legal Definition:** Finalized corpus-agnostic from external regulatory and statutory authority only. Full taxonomy in `CLAUDE.md`. E = GHG Protocol/EPA/SEC; S = NLRA/FLSA/Title VII/UFLPA/UN Guiding Principles; G = ERISA § 404/DOL 29 CFR § 2550.404a-1/SEC/SOX; Sustainability = UN Brundtland (1987) cross-cutting modifier only; Non-ESG = no nexus to E, S, G, or sustainability disclosures.

**Sustainability label (resolved April 22, 2026):** Cross-cutting modifier. Binary `is_sustainability` flag in `esg_corpus_labels.csv`. Primary classifier outputs are E, S, G, Non-ESG — four classes only.

**Legacy corpus (resolved April 22, 2026):** `esg_master_corpus_dedup_classified/` deleted.

**Label construction (resolved April 22, 2026):** argmax(E, S, G); tiebreaker G > E > S; zero-signal → Non-ESG.
- G 171 (38.5%) | Non-ESG 134 (30.2%) | E 92 (20.7%) | S 47 (10.6%)
- `is_sustainability`: 64 cases (14.4%) | `is_greenwash`: 1 case
- Class imbalance: SMOTE or class-weighted loss required for S and greenwash.

**Leakage masking (resolved and verified April 22, 2026):** 14 terms masked as `[OUTCOME]`. 3,028 masks across 422/444 files. Zero ALL-CAPS survivors. Clean.

**Token audit (April 22, 2026):** 99.1% of docs (440/444) exceed 512 tokens. Average 8,824 tokens/case. DistilBERT/RoBERTa 512-token limit unworkable without chunking. Longformer (4,096 tokens) or sliding-window chunking required — must address in methodology slide.

**Greenwash flag:** Noise filter reduced greenwash cases from 3 (expected) to 1 — 2 false negatives. Address in limitations slide.

**Tool strategy:** Claude.ai for planning and slide drafting. Claude Code for execution. VSC for code editing and Git.

---

## 3. Specific Next Steps

**~~Step 1 — Sustainability label~~** ✓ Cross-cutting modifier.
**~~Step 2 — Legacy corpus~~** ✓ Deleted.
**~~Step 3 — Verify environment~~** ✓ All paths resolve.
**~~Step 4 — Preprocessing pipeline~~** ✓ Scripts 01–03 complete.
**~~Step 5 — Label construction~~** ✓ `esg_corpus_labels.csv` produced.
**~~Step 5b — Text cleaning / leakage / token audit~~** ✓ `ESG_corpus_cleaned_v1.csv` produced. **Verify masking next session.**

**~~Step 6 — Verify [OUTCOME] masking~~** ✓ Complete April 22, 2026. 3,028 masks across 422/444 files. Zero ALL-CAPS leakage survivors. 22 unmasked files are Non-ESG (no leakage terms expected).

**~~Step 7a — Write modeling scripts~~** ✓ Complete April 23, 2026. `06_esg_ml_baseline.py`, `07_esg_longformer.py`, `08_esg_xai_visualizations.py` written. `config.py` updated with new output paths and manifests 06–08.

**~~Step 7b — Draft Methodology + Limitations slide narrative~~** ✓ Complete April 23, 2026. `esg_slide_narrative.md` — narrative prose, methodology table, 16 inline citations, no bullets.

**~~Step 8a — Execute 06_esg_ml_baseline.py (v1)~~** ✓ Superseded April 23, 2026. Original results: RF Macro-F1 = 0.6078, XGBoost Macro-F1 = 0.8253. Prompt 2 reopened — script revised.

**~~Step 8a-rev — Execute revised 06_esg_ml_baseline.py~~** ✓ Complete April 23, 2026. Split: 310/67/67 (stratified). 5 models: Majority Baseline F1=0.1398, BoW-LR F1=0.7529, LR F1=0.7409, ComplementNB F1=0.6117, RF F1=0.6760, XGBoost F1=0.7684 (authoritative). 27 plots: 5 confusion matrices, 4 per-pillar ROC images, 2 feature importance, 2 global SHAP bar charts, 8 per-pillar dot beeswarms (RF+XGBoost×4 pillars), 2 waterfall (E pillar, RF+XGBoost). SHAP uses new Explanation-object API (`explainer(X)`). [OUTCOME] masking verified clean.

**~~Step 10 — Execute 10_esg_descriptive_analysis.py~~** ✓ Complete April 23, 2026. Word clouds (global + 4 pillars + composite) and per-pillar n-gram charts (top-20 unigrams/bigrams/trigrams × 4 pillars, one image each) produced. 11 PNGs + CSV in `esg_corpus_outputs/descriptive_analysis/`. `10_manifest.json` written.

**~~Step 10b — Reproducibility zip (snapshot_10_descriptive_analysis_v1_4232026.zip)~~** ✓ Complete April 23, 2026. 11.7 MB zip in `step_zips/`. Contains 11 PNGs, ngrams CSV, snapshot CSV+JSON, script, manifests, CLAUDE.md, status.md. SHA-256: `eb39099fada1e8f999a8036b53bac293325bf7ab47cb2d8574b37be17e518c90`. Frozen anchor verified: `a2b95dfd...` (matches master).

**Step 8b — Execute 07_esg_longformer.py (Colab GPU required).** Script revised April 23, 2026: added AMP mixed precision (FP16), gradient checkpointing, fold-level resume via `07_progress.json`, training loss logging, MCC per epoch, 3-seed x 3-fold config. Run in Colab T4/A100. On each session start, re-run the same cell -- it skips completed folds automatically. Session 1 target: seed 42 (all 3 folds, ~4-6 hrs T4). Session 2: seeds 123 + 7. Produces `longformer_s{seed}_f{fold}.pt` checkpoints + `longformer_embeddings.npy` + `07_manifest.json`.

**Step 8c — Execute 08_esg_xai_visualizations.py** (after 07 completes). Produces ROC curves, word clouds, SHAP, t-SNE, attention heatmaps.

**~~Step 8d — Reproducibility package + version control~~** ✓ Complete April 23, 2026. 6 versioned snapshot pairs (CSV + JSON) in `snapshots/`. 6 per-step zips in `step_zips/` (each contains snapshot CSV + JSON + manifest + script + step outputs). Jupyter notebook included in `snapshot_06_ml_baseline.zip`. Git repo initialized; 2 commits pushed to GitHub (c1238c4, 8f8a4e7). `ESG_corpus_cleaned_v1.csv` git-excluded (24MB) but included in `snapshot_05_text_cleaning.zip`. PAT deleted — generate new one before next push.

**~~Step 8e — Regenerate step-06 reproducibility zip (revised 27-plot outputs)~~** ✓ Complete April 23, 2026. `06_create_reproducibility_zip.py` executed. 23.6 MB zip in `step_zips/`. Contains 24 PNGs, 5 .pkl models, predictions CSV, metrics JSON, snapshot CSV+JSON, script, manifests. Frozen anchor verified: `a2b95dfd...` (matches master). SHA-256: `b7b3e40e1544dd52d7d79da6521bb285db7a6f7ad5fbe35ffb9a9d3037104fef`.

**Step 9 — Revise all 9 PowerPoint slides.** Use `esg_slide_narrative.md` as source. Narrative prose, no bullets, inline citations, methodology table + narrative, descriptive-to-predictive pipeline explicit, expanded limitations. Framing: surfacing ESG legal issues from case law. Produce in Claude.ai using pptx skill.

---

## 4. Versioning & SHA-256 Integrity — STANDING REQUIREMENTS

**Versioning scheme:** All snapshot files, zips, and frozen datasets carry `_v1_4232026` suffix. Increment `PACKAGE_VERSION` in `config.py` when pipeline decisions change. Never produce a versioned artifact without the suffix.

**Auto-update rule:** status.md and CLAUDE.md must be updated immediately after every completed task — without user request. esg_prompts.md must be updated when any prompt changes status or produces new outputs. SHA-256 hashes must be recorded in all four locations: snapshot JSON, prompt entry, CLAUDE.md Resolved Decisions, and the reproducibility_manifest inside the step zip.

**Master SHA-256 anchors (v1 — April 23, 2026):**

| File | SHA-256 |
|---|---|
| `feature_matrix_v1_frozen.csv` | `a2b95dfd616fc8573c1f27a4d4a4b18c02136c011a9ded6772b28bb75a00283e` |
| `esg_corpus_labels.csv` | `97ee91e30aa008e46940bd189130714d213be3cbc86023f3ba615a5d441a6efd` |
| `snapshot_01_deduplication_v1_4232026.csv` | `ae7f5a54d8ae4b8e93dc23b6b609cd4c568e7c8732a20fa55a5e4814461e2af3` |
| `snapshot_02_noise_filtering_v1_4232026.csv` | `d67fbb4edff9a39c4c6d1d9c13fda2ae55c68e91da098774744dc54491be5a59` |
| `snapshot_03_corpus_stats_v1_4232026.csv` | `75cc74176bbf56f8d90fc6e1b907b98e1bcd202289c95b53a684531ac9494fe0` |
| `snapshot_04_label_construction_v1_4232026.csv` | `97ee91e30aa008e46940bd189130714d213be3cbc86023f3ba615a5d441a6efd` |
| `snapshot_05_text_cleaning_v1_4232026.csv` | `aaa9f821b49cb1c5cb007453f381340ac473e2758d99becfec887db4a76c1c6e` |
| `snapshot_06_ml_baseline_v1_4232026.csv` | `b3f0ceb6be16cf57524f5b931280d9fea6d1297e72cd8f1bc3c0594bf83add4d` *(regenerated from revised outputs)* |
| `snapshot_06_ml_baseline_v1_4232026.zip` | `b7b3e40e1544dd52d7d79da6521bb285db7a6f7ad5fbe35ffb9a9d3037104fef` |
| `esg_slide_narrative.md` | `d0aa3af748504d42c4227c7284f28b69ee55a8d55df8e8de7df14bf502fba707` |
| `ngrams_E_v1_4232026.png` | *(verify on next run — supersedes combined ngrams_all_pillars image)* |
| `ngrams_S_v1_4232026.png` | *(verify on next run)* |
| `ngrams_G_v1_4232026.png` | *(verify on next run)* |
| `ngrams_NonESG_v1_4232026.png` | *(verify on next run)* |
| `ngrams_v1_4232026.csv` | `7bcee41fe22c43a45a41e7a18df037cd78f709890f0fd0fda8c1627cf8c7077f` |
| `wordcloud_global_v1_4232026.png` | `00115d0d51c46d15c5ec7159953ca9bb15f5d38a62cb24b1f62419746f349219` |
| `wordcloud_composite_pillars_v1_4232026.png` | `77a070fab8c88cf8b27c59c3f1c479e655a399524e9ad7b2923cd97828fc30c5` |
| `snapshot_10_descriptive_analysis_v1_4232026.zip` | `eb39099fada1e8f999a8036b53bac293325bf7ab47cb2d8574b37be17e518c90` |

---

## 5. The No-Bullet/Narrative Rule — DO NOT FORGET

Every revised slide must be written in narrative prose sentences — no bullet points, no arrow lists, no fragments. Each slide reads as one or two coherent connected paragraphs. The sole exception is the methodology table. This rule applies to every slide without exception.

---

## 6. Files

| File | Location | Description |
|---|---|---|
| `CLAUDE.md` | Project root | Rules, taxonomy, revision guidelines |
| `status.md` | Project root | This file |
| `config.py` | Project root | All project paths — run to verify environment |
| `01_esg_deduplicate.py` | Project root | Deduplication — 918 canonical / 364 dupes |
| `02_non_esg_filter_noise.py` | Project root | Noise filtering — 444 retained / 474 excluded |
| `03_esg_corpus_stats.py` | Project root | Descriptive stats + pillar metadata CSV |
| `04_esg_label_construction.py` | Project root | Label assignment — `esg_corpus_labels.csv` |
| `05_esg_text_clean.py` | Project root | Text cleaning, leakage masking, token audit |
| `06_esg_ml_baseline.py` | Project root | TF-IDF + RF + XGBoost + SHAP → `esg_corpus_outputs/ml_baseline/` |
| `07_esg_longformer.py` | Project root | Longformer fine-tuning (Colab GPU) → `esg_corpus_outputs/longformer/` |
| `08_esg_xai_visualizations.py` | Project root | ROC, word clouds, SHAP, t-SNE, attention → `esg_corpus_outputs/visualizations/` |
| `09_create_reproducibility_package.py` | Project root | Generates snapshots/, step_zips/, Jupyter notebook |
| `10_esg_descriptive_analysis.py` | Project root | Word clouds + n-gram extraction → `esg_corpus_outputs/descriptive_analysis/` |
| `06_create_reproducibility_zip.py` | Project root | Step-06 reproducibility zip → `step_zips/snapshot_06_ml_baseline_v1_4232026.zip` (revised 27-plot outputs) |
| `10_create_reproducibility_zip.py` | Project root | Step-10 reproducibility zip → `step_zips/snapshot_10_descriptive_analysis_v1_4232026.zip` |
| `esg_slide_narrative.md` | Project root | Methodology + Limitations slide prose with 16 inline citations |
| `esg_prompts.md` | Project root | Canonical prompt list for all pipeline phases |
| `esg_corpus_outputs/esg_corpus_labels.csv` | Outputs | Authoritative label source |
| `esg_corpus_outputs/ESG_corpus_cleaned_v1.csv` | Outputs | Authoritative model input (24MB — git-excluded, in step_05 zip) |
| `esg_corpus_outputs/ml_baseline/` | Outputs | RF + XGBoost models (.pkl), SHAP PNGs, metrics JSON, predictions CSV |
| `esg_corpus_filtered/esg_corpus_pillar_metadata.csv` | Filtered | Per-file pillar scores + flags |
| `esg_corpus_filtered/esg_corpus_stats.txt` | Filtered | Descriptive statistics report |
| `snapshots/` | Project root | 6 × (CSV + JSON) versioned step snapshots |
| `step_zips/` | Project root | 6 per-step reproducibility zips |
| `ESG_Litigation_Classifier_Reproducibility.ipynb` | `reproducibility_package/` | Full pipeline Jupyter notebook |
| Corpus zip | Re-upload to Claude.ai if needed | `001-BroadcastMusicIncvTexBorderManagementInc.zip` |
| Assignment | Re-upload to Claude.ai if needed | `Spring_2026_DL_Team_Project_Proposal.docx` |
| Marked-up proposal | Re-upload to Claude.ai if needed | `ESGProjeectMarkedUp.pdf` |

---

## 7. Canonical Naming Reference

| Entity | Correct name |
|---|---|
| Raw corpus folder | `esg_corpus/` |
| Deduped corpus folder | `esg_corpus_deduped/` |
| Duplicates subfolder | `esg_corpus_dupes/` |
| Filtered corpus folder | `esg_corpus_filtered/` |
| Excluded subfolder | `esg_corpus_excluded/` |
| Outputs folder | `esg_corpus_outputs/` |
| Filter log | `esg_filter_log.csv` |
| Pillar metadata CSV | `esg_corpus_pillar_metadata.csv` |
| Stats report | `esg_corpus_stats.txt` |
| Labels CSV | `esg_corpus_labels.csv` |
| Cleaned model input | `ESG_corpus_cleaned_v1.csv` |
| Step 1 manifest | `01_manifest.json` |
| Step 2 manifest | `02_manifest.json` |
| Step 3 manifest | `03_manifest.json` |
| Step 4 manifest | `04_manifest.json` |
| Step 5 manifest | `05_manifest.json` |
| Step 6 manifest | `06_manifest.json` |
| Step 9 manifest | `09_manifest.json` |
| Deduplication snapshot | `snapshots/snapshot_01_deduplication.{csv,json}` |
| Noise filtering snapshot | `snapshots/snapshot_02_noise_filtering.{csv,json}` |
| Corpus stats snapshot | `snapshots/snapshot_03_corpus_stats.{csv,json}` |
| Label construction snapshot | `snapshots/snapshot_04_label_construction.{csv,json}` |
| Text cleaning snapshot | `snapshots/snapshot_05_text_cleaning.{csv,json}` |
| ML baseline snapshot | `snapshots/snapshot_06_ml_baseline.{csv,json}` |
| GitHub repo | `https://github.com/mkiernan4-MSAIB/esg_litigation_corpus_nlp_ml_dp_experiment` |
