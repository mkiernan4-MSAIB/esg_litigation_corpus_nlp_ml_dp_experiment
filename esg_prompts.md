The following is the comprehensive list of all project prompts, updated to reflect the **complete status of the preprocessing pipeline** and the **open status of the modeling and synthesis phases**. Each prompt is tailored to the refined **444-case high-signal corpus** and the mandatory transition to the **Longformer architecture**.

### ** Prompt 1: Aggressive Preprocessing & Deduplication (Status: COMPLETE)**

"I have a corpus of **1,282 litigation documents** (918 unique) stored as text files. Please write Python code to:
1.  **Deduplicate:** Identify exact duplicates and near-duplicates using **cosine similarity > 0.95**.
2.  **Clean:** Remove case captions, Westlaw headers, and reporter citations.
3.  **Noise Filtering:** Flag the 412 'Copyright' cases as noise for removal.
4.  **Outcome Leakage:** Remove 14 outcome-indicating words (dismissed, granted, etc.) using context-aware regex.
5.  **Token Audit:** Report the percentage of documents exceeding **512 tokens**.
Save the output as `ESG_corpus_cleaned_v1.csv`."

**Technical Note:** This stage is finalized. The audit confirmed that **99.1% of cases** exceed the 512-token limit, and **3,028 [OUTCOME] masks** have been successfully applied to 422 files.

**SHA-256 integrity anchors (v1 — April 23, 2026):**
- `snapshot_01_deduplication_v1_4232026.csv` → `ae7f5a54d8ae4b8e93dc23b6b609cd4c568e7c8732a20fa55a5e4814461e2af3`
- `snapshot_02_noise_filtering_v1_4232026.csv` → `d67fbb4edff9a39c4c6d1d9c13fda2ae55c68e91da098774744dc54491be5a59`
- `snapshot_03_corpus_stats_v1_4232026.csv` → `75cc74176bbf56f8d90fc6e1b907b98e1bcd202289c95b53a684531ac9494fe0`
- `snapshot_04_label_construction_v1_4232026.csv` → `97ee91e30aa008e46940bd189130714d213be3cbc86023f3ba615a5d441a6efd`
- `snapshot_05_text_cleaning_v1_4232026.csv` → `aaa9f821b49cb1c5cb007453f381340ac473e2758d99becfec887db4a76c1c6e`
- `feature_matrix_v1_frozen.csv` (master anchor) → `a2b95dfd616fc8573c1f27a4d4a4b18c02136c011a9ded6772b28bb75a00283e`
- `esg_corpus_labels.csv` → `97ee91e30aa008e46940bd189130714d213be3cbc86023f3ba615a5d441a6efd`

---

### ** Prompt 2: Machine Learning Baseline & Interpretability (Status: COMPLETE — April 23, 2026)**

"I require Python code for a Google Colab environment to establish a Machine Learning baseline using the authoritative `ESG_corpus_cleaned_v1.csv` and `esg_corpus_labels.csv`. 

The pipeline must:
1.  **Feature Engineering:** Implement **TF-IDF vectorization** with `min_df=3` and N-grams (1,2) on the [OUTCOME] masked text.
2.  **Multi-Class Strategy:** Train a **Random Forest** and **XGBoost** classifier for the four primary classes: **Environmental, Social, Governance, and Non-ESG**.
3.  **Imbalance Mitigation:** Apply `class_weight='balanced'` to address the 'Social' pillar, which represents only **10.6% of the corpus** (47 cases).
4.  **Evaluation:** Report **Macro-F1 and Matthews Correlation Coefficient (MCC)**.
5.  **Interpretability:** Use **SHAP TreeExplainer** to produce high-resolution beeswarm and waterfall plots for the top 20 features."

**Technical Note:** Classical ML is required as a baseline because the unique corpus size (444 cases) is below the threshold where Deep Learning consistently outperforms shallow models.

**Execution results (April 23, 2026):** `06_esg_ml_baseline.py` executed locally. RF Macro-F1 = 0.6078 / MCC = 0.5721. XGBoost Macro-F1 = 0.8253 / MCC = 0.8015. XGBoost is authoritative ML baseline. 3 SHAP plots saved to `esg_corpus_outputs/ml_baseline/`.

**SHA-256 integrity anchors (v1 — April 23, 2026):**
- `snapshot_06_ml_baseline_v1_4232026.csv` → `590b017fe1c73bd33cac1f52fc7373d225c21fd50a9c386ac074568636eae5a9`
- `feature_matrix_v1_frozen.csv` (training input anchor) → `a2b95dfd616fc8573c1f27a4d4a4b18c02136c011a9ded6772b28bb75a00283e`

---

### ** Prompt 3: Longformer Fine-Tuning & Regularization (Status: SCRIPT WRITTEN — pending Colab GPU execution)**

"I need to fine-tune a **Longformer** model to categorize ESG legal issues, as the recent token audit confirmed that **99.1% of my 444 cases exceed the 512-token limit**, with an average length of **8,824 tokens per case**.

The training code must:
1.  **Architecture:** Utilize **`allenai/longformer-base-4096`** to accommodate the ultra-long judicial opinions, using a sliding window or global attention for the first 4,096 tokens.
2.  **Regularization:** Freeze the **bottom 8 layers** of the model to prevent catastrophic forgetting on this research-scale dataset of 444 cases.
3.  **Loss Function:** Implement a **Class-Weighted Cross-Entropy Loss** to penalize errors on the sparse 'Social' (47 cases) and 'Greenwash' (1 case) labels.
4.  **Hyperparameters:** Use a learning rate of **2e-5** with a **linear warmup** over the first 10% of steps and a batch size appropriate for 16GB VRAM.
5.  **Stability:** Execute the training across **3 random seeds** and report the mean and standard deviation of the **Macro-F1 score**."

**Technical Note:** The shift from RoBERTa to Longformer is a direct consequence of the finding that standard transformer limits are unworkable for this specific litigation corpus.

**Script:** `07_esg_longformer.py` written April 23, 2026. Run in Colab T4/A100 runtime. Produces checkpoints in `esg_corpus_outputs/longformer/` and CLS embeddings for Prompt 4 visualizations.

**SHA-256:** To be recorded upon Colab execution. Hash `snapshot_07_longformer_v1_{DATE}.csv`, best checkpoint `.pt` file, and `longformer_embeddings.npy` — add all to CLAUDE.md, status.md, and this prompt entry.

---

### ** Prompt 4: Advanced XAI & Visualizations (Status: SCRIPT WRITTEN — pending Prompt 3 execution)**

"Please write plotting code for **Phase 7: Results Synthesis** using the results from the ML and Longformer models. 

The script must produce:
1.  **Overlaid ROC Curves:** A comparison of all models across the four ESG pillars.
2.  **Word Clouds:** Generate clouds of **top terms per class label** to visualize doctrinal differences.
3.  **SHAP PartitionExplainer:** Apply this to the Longformer to identify substantive legal triggers (e.g., **ERISA § 404** or **UFLPA**).
4.  **t-SNE Plot:** A 2D projection of the **Longformer document embeddings**, colored by outcome class.
5.  **Attention Heatmaps:** Visualize where the Longformer focuses its attention relative to the [OUTCOME] masks."

**Technical Note:** This phase validates that the model is learning substantive legal theories (surfacing issues) rather than memorizing case-specific noise.

**Script:** `08_esg_xai_visualizations.py` written April 23, 2026. Run after `07_esg_longformer.py` completes. Gracefully skips Longformer-dependent plots if embeddings not yet available. Outputs to `esg_corpus_outputs/visualizations/`.

**SHA-256:** To be recorded upon execution. Hash all PNG outputs and add to CLAUDE.md, status.md, and this prompt entry.

---

### ** Prompt 5: Narrative Proposal Revision (Status: COMPLETE — April 23, 2026)**

"Draft the narrative prose for the **Methodology and Limitations slides** of the project proposal, adhering to the **no-bullet rule**. 

The prose must:
1.  **Explain Pipeline:** Describe the transition from the 1,282 raw files to the 444 high-signal cases.
2.  **Justify Longformer:** Explain the transition to **Longformer architecture** as a data-driven response to the 99.1% token exceedance.
3.  **Integrity:** Detail how **[OUTCOME] masking** of 14 terms prevented data leakage.
4.  **Citations:** Include inline numbered citations for taxonomy anchors like **UN Brundtland (1987)** and **SEC Rule 10b-5**."

**Technical Note:** This prompt addresses specific instructor feedback from the initial proposal regarding formatting and the need for connected narrative sections.

**Output (April 23, 2026):** `esg_slide_narrative.md` — full narrative prose for Methodology (table + two paragraphs) and Scope/Limitations slides. 16 inline citations. No bullets. Ready for PowerPoint production in Claude.ai using the pptx skill.

**SHA-256 integrity anchor (v1 — April 23, 2026):**
- `esg_slide_narrative.md` → `d0aa3af748504d42c4227c7284f28b69ee55a8d55df8e8de7df14bf502fba707`