# CLAUDE.md — ESG Litigation Classifier
## AIGB 7290 | Huang, Kiernan, Sooknanan
## Auto-loaded by Claude Code. Paste manually in Claude.ai and IDE sessions.

---

## Response Style

- **NO FLUFF.** No "I understand," "Certainly," "Great question," or "Here is the summary."
- **Narrative over bullets.** Use prose paragraphs for all explanations, analyses, and slide content. Bullets only if explicitly requested for a list (max 5 items).
- **Academic tone.** High-level legal and technical vocabulary throughout.
- **Citations.** Inline numbered references [#] for every factual claim. No footnotes.
- **Code.** Concise diffs or specific functions only — never reprint entire files.
- **Prohibited words.** Never use "straightforward," "boundaries," "I cannot assist," or "I need to clarify."
- **status.md + CLAUDE.md updates.** Update both files iteratively throughout every session — after each completed decision, deliverable, or meaningful step. Do not wait until end of session.

---

## Project Context

**Course:** AIGB 7290 Deep Learning — Fordham University
**Goal:** Classify 1,282 Westlaw case documents into E, S, G, or Non-ESG using NLP and deep learning. Sustainability is a cross-cutting modifier (`is_sustainability` binary flag), not a standalone class.
**Tech stack:** Python, Transformers (Legal-BERT preferred; Longformer required due to token length — see below), LDA topic modeling, SHAP/LIME explainability.
**Corpus:** 1,282 raw OCR markdown files from Westlaw. 918 unique after deduplication. 444 after noise filtering. Cleaned model input: `ESG_corpus_cleaned_v1.csv`.
**Pipeline order:** Descriptive analysis (LDA/topic modeling → clustering) → Predictive analysis (classification → SHAP/LIME).
**Grade received:** 89/100. Full proposal revision required before execution. No slides revised yet.

---

## Tool Strategy

**Claude.ai** — planning, decisions, document review, slide drafting, taxonomy design, and producing artifacts. Maintains `CLAUDE.md` and `status.md`.
**Claude Code** — execution. Runs Python scripts, processes corpus, updates `status.md` after each completed step. `CLAUDE.md` auto-loaded from project root.
**VSC** — code editing, syntax review, Git version control.

---

## Project Root

Windows path : `C:\GoogleDriveProfiles\Michael_Fordham\Shared drives\ESG DL Project\esg_project`
Google Drive : `/content/drive/Shared Drives/ESG DL Project/esg_project`

---

## Project File Structure

```
esg_project/
├── CLAUDE.md
├── status.md
├── config.py                              # single source of truth for all paths
├── 01_esg_deduplicate.py
├── 02_non_esg_filter_noise.py
├── 03_esg_corpus_stats.py
├── 04_esg_label_construction.py
├── 05_esg_text_clean.py
├── 01_manifest.json
├── 02_manifest.json
├── 03_manifest.json
├── 04_manifest.json
├── 05_manifest.json
├── esg_corpus/                            # 1,282 raw .md files
├── esg_corpus_deduped/                    # 918 unique canonical cases
│   └── esg_corpus_dupes/                  # 364 quarantined duplicates
├── esg_corpus_filtered/                   # 444 ESG-relevant cases
│   ├── esg_corpus_excluded/               # 474 quarantined non-ESG noise
│   ├── esg_filter_log.csv
│   ├── esg_corpus_pillar_metadata.csv     # per-file pillar scores + flags
│   └── esg_corpus_stats.txt
└── esg_corpus_outputs/
    ├── esg_corpus_labels.csv              # authoritative label source
    └── ESG_corpus_cleaned_v1.csv          # authoritative model input
```

---

## Classification Taxonomy

All classification decisions must adhere strictly to the following corpus-agnostic legal definitions, built exclusively from external regulatory and statutory authority.

### Environmental (E)
Corporate conduct affecting the natural environment via direct or value-chain operations. Measured across GHG Protocol Scope 1 (direct emissions), Scope 2 (purchased energy), and Scope 3 (value chain). Legal triggers include pollution, resource depletion, biodiversity harm, greenwashing, and failure to disclose material climate-related financial risk. Regulatory anchors: EPA (Clean Air Act, Clean Water Act, CERCLA), SEC climate disclosure rules (2024), California SB 253/261, EU CSRD/Taxonomy.

### Social (S)
Corporate conduct affecting human capital, communities, and supply chain integrity. Legal triggers include labor rights and fair wages, worker safety, supply chain accountability, forced labor (Uyghur Forced Labor Prevention Act 2021), human rights at international supplier sites, DEI, consumer protection, and corporate political spending. Regulatory anchors: NLRA, FLSA, Title VII, OSHA, UN Guiding Principles on Business and Human Rights (2011), EU CSDDD (2024).

### Governance (G)
Structures and mechanisms for corporate direction, control, and fiduciary accountability. Legal triggers include board composition and independence, executive compensation, shareholder rights, proxy voting, fiduciary duty (ERISA § 404 — duties of prudence and loyalty), anti-corruption (FCPA), cybersecurity governance, and ESG disclosure accuracy. Regulatory anchors: SEC proxy rules and 10-K/Q disclosure requirements, DOL Final ESG Rule 29 CFR § 2550.404a-1 (2022), Dodd-Frank Act, Sarbanes-Oxley, Delaware corporate law.

### Sustainability (Cross-Cutting Modifier — NOT a standalone class)
**Decision (April 22, 2026):** Binary `is_sustainability` flag in `esg_corpus_labels.csv`. Every case receives a primary label (E, S, G, or Non-ESG). Flag = 1 when legal theory turns on long-term viability commitments materially inconsistent with actual operations, per UN Brundtland Commission (1987). No primary legal authority creates a freestanding sustainability cause of action.

### Non-ESG (Exclusion)
Any case with no nexus to environmental conduct, treatment of people, governance structures, or sustainability disclosures — regardless of whether "ESG" appears in the text.

---

## Resolved Decisions

**Sustainability label (April 22, 2026):** Cross-cutting modifier. Four primary classes only: E, S, G, Non-ESG.

**Legacy corpus (April 22, 2026):** `esg_master_corpus_dedup_classified/` deleted.

**Label construction (April 22, 2026):** argmax(E, S, G); tiebreaker G > E > S; zero-signal → Non-ESG. Distribution: G 171 (38.5%), Non-ESG 134 (30.2%), E 92 (20.7%), S 47 (10.6%). `is_sustainability`: 64 cases. `is_greenwash`: 1 case.

**Leakage masking (April 22, 2026 — verified):** 14 terms masked as `[OUTCOME]` — dismissed, granted, affirmed, reversed, remanded, vacated, denied, overruled, enjoined, prevailed, affirming, reversing, remanding, vacating. Context-aware regex + ALL-CAPS standalone forms. Verified: 3,028 masks across 422/444 files. Zero ALL-CAPS leakage survivors. 22 files with no masks are Non-ESG cases (expected).

**Token audit (April 22, 2026):** 99.1% of docs exceed 512 tokens. Average 8,824 tokens/case. DistilBERT/RoBERTa 512-token limit unworkable. Longformer (4,096 tokens) or sliding-window chunking required. Must be addressed explicitly in methodology slide.

**Class imbalance (April 22, 2026):** S at 10.6%, greenwash at 1 case. SMOTE or class-weighted loss required. 2 greenwash cases lost as false negatives in noise filter — address in limitations slide.

**Reproducibility package (April 23, 2026):** `snapshots/` — 6 versioned snapshot CSV+JSON pairs named by step (snapshot_01_deduplication through snapshot_06_ml_baseline). `step_zips/` — 6 per-step zips, each self-contained with snapshot CSV+JSON, manifest, script, and step-specific outputs (SHAP PNGs in step 06, full cleaned CSV in step 05). Jupyter notebook in step 06 zip. Git initialized; 2 commits pushed to GitHub. PAT deleted April 23, 2026 — generate new one before next push. GitHub: https://github.com/mkiernan4-MSAIB/esg_litigation_corpus_nlp_ml_dp_experiment

**ML baseline results (April 23, 2026 — executed):** Random Forest Macro-F1 = 0.6078 / MCC = 0.5721 (S recall 0.13 — RF underperforms on sparse high-dimensional features with minority class). XGBoost Macro-F1 = 0.8253 / MCC = 0.8015 (S recall 0.60). XGBoost is the authoritative ML baseline. SHAP plots saved to `esg_corpus_outputs/ml_baseline/`. `06_manifest.json` written.

---

## Proposal Revision Rules (Professor Feedback — 89/100)

Every revised slide must comply with all of the following:

**No bullets.** The professor explicitly marked "too many bullets" across multiple slides. All content must be written in narrative prose sentences — no bullet points, no arrow lists, no fragments. Each slide reads as one or two coherent connected paragraphs. The sole exception is the methodology table, which the professor suggested as the structural anchor, after which narrative prose must immediately follow.

**Inline citations on every slide.** Every factual claim requires a numbered inline citation [#]. References slide at the end.

**Methodology table + narrative.** Table outlines phases; prose describes each phase in connected sentences after the table.

**Explicit descriptive-to-predictive pipeline.** Must show: topic modeling/LDA → clustering → classification → SHAP/LIME explainability.

**Expanded scope and limitations.** Must include: bias, hallucination risk, reproducibility, sample size constraints, token length constraint (99.1% exceed 512 tokens), and class imbalance (S underrepresented, greenwash = 1 case).

**Higher vocabulary.** Professor marked "higher language / vocabulary consistency" on the significance slide.

**Framing.** Project must be framed as surfacing ESG legal issues from the lens of case law — what companies are concerned about and why — not as a greenwashing study.

**Architecture note for methodology slide.** Longformer or sliding-window chunking must be explained as the consequence of the 99.1% token exceedance finding, not as an arbitrary model choice.
