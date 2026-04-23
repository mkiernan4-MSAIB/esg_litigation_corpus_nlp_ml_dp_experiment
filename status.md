# Session Status — AIGB 7290 ESG Deep Learning Project
**Date:** April 23, 2026 | **Team:** James Huang, Michael Kiernan, Amanda Sooknanan

---

## 1. Current Task Status

**Session April 23, 2026 — MODELING + VISUALIZATION SCRIPTS WRITTEN. SLIDE NARRATIVE DRAFTED.**

Scripts 06–08 written and ready for Colab execution. Slide narrative prose for Methodology and Limitations slides drafted in `esg_slide_narrative.md`. PowerPoint revision (9 slides) still pending — narrative content is ready; deck production in Claude.ai is the next step.

**EXECUTION ORDER:**
1. Run `06_esg_ml_baseline.py` locally or in Colab (CPU-viable). Produces RF + XGBoost + SHAP outputs.
2. Run `07_esg_longformer.py` in Colab with GPU runtime (T4/A100 required). Produces Longformer checkpoints + embeddings.
3. Run `08_esg_xai_visualizations.py` after both above complete. Produces all 5 visualization outputs.
4. Produce PowerPoint deck in Claude.ai using `esg_slide_narrative.md` as content source.

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

**~~Step 8a — Execute 06_esg_ml_baseline.py~~** ✓ Complete April 23, 2026. RF Macro-F1 = 0.6078 / MCC = 0.5721. XGBoost Macro-F1 = 0.8253 / MCC = 0.8015. 3 SHAP plots saved. `06_manifest.json` written.

**Step 8b — Execute 07_esg_longformer.py (Colab GPU required).** Run in Colab T4/A100. Produces Longformer checkpoints + CLS embeddings.

**Step 8c — Execute 08_esg_xai_visualizations.py** (after 07 completes). Produces ROC curves, word clouds, SHAP, t-SNE, attention heatmaps.

**Step 9 — Revise all 9 PowerPoint slides.** Use `esg_slide_narrative.md` as source. Narrative prose, no bullets, inline citations, methodology table + narrative, descriptive-to-predictive pipeline explicit, expanded limitations. Framing: surfacing ESG legal issues from case law. Produce in Claude.ai using pptx skill.

---

## 4. The No-Bullet/Narrative Rule — DO NOT FORGET

Every revised slide must be written in narrative prose sentences — no bullet points, no arrow lists, no fragments. Each slide reads as one or two coherent connected paragraphs. The sole exception is the methodology table. This rule applies to every slide without exception.

---

## 5. Files

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
| `esg_slide_narrative.md` | Project root | Methodology + Limitations slide prose with 16 inline citations |
| `esg_corpus_outputs/esg_corpus_labels.csv` | Outputs | Authoritative label source |
| `esg_corpus_outputs/ESG_corpus_cleaned_v1.csv` | Outputs | Authoritative model input |
| `esg_corpus_filtered/esg_corpus_pillar_metadata.csv` | Filtered | Per-file pillar scores + flags |
| `esg_corpus_filtered/esg_corpus_stats.txt` | Filtered | Descriptive statistics report |
| Corpus zip | Re-upload to Claude.ai if needed | `001-BroadcastMusicIncvTexBorderManagementInc.zip` |
| Assignment | Re-upload to Claude.ai if needed | `Spring_2026_DL_Team_Project_Proposal.docx` |
| Marked-up proposal | Re-upload to Claude.ai if needed | `ESGProjeectMarkedUp.pdf` |

---

## 6. Canonical Naming Reference

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
