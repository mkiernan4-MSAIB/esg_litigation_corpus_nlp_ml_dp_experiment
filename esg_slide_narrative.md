# ESG Slide Narrative — Proposal Revision
# AIGB 7290 | Huang, Kiernan, Sooknanan | Fordham University
# Drafted per professor feedback: narrative prose, no bullets, inline citations, high vocabulary.

---

## Slide: Methodology

| Phase | Method | Output |
|---|---|---|
| 1 — Corpus Preparation | Deduplication (cosine sim > 0.95), noise filtering, OCR normalization | 444 high-signal cases from 1,282 raw Westlaw documents |
| 2 — Label Construction | argmax(E, S, G pillar scores); tiebreaker G > E > S; zero-signal → Non-ESG | esg_corpus_labels.csv: 4 primary classes + is_sustainability flag |
| 3 — Leakage Prevention | Context-aware regex masking of 14 outcome-indicating terms as [OUTCOME] | 3,028 masks across 422/444 files; zero ALL-CAPS leakage survivors |
| 4 — Descriptive Analysis | LDA topic modeling, K-Means clustering on TF-IDF embeddings | Latent doctrinal themes per ESG pillar |
| 5 — ML Baseline | TF-IDF + Random Forest + XGBoost with class-weighted loss; SHAP TreeExplainer | Macro-F1 / MCC benchmarks; top-20 legal trigger features |
| 6 — Deep Learning | Longformer-base-4096 fine-tuning; 3 seeds × 5-fold CV; frozen bottom-8 layers | Mean ± SD Macro-F1 across 15 training runs |
| 7 — Explainability | SHAP PartitionExplainer, t-SNE embeddings, attention heatmaps | Doctrinal attribution maps; [OUTCOME] mask attention validation |

The analytical pipeline proceeds in two sequential phases — descriptive then predictive — to ensure that latent structural patterns in the corpus are established before supervised classification is attempted [1]. In the descriptive phase, Latent Dirichlet Allocation identifies co-occurring legal terms across documents and assigns each case a probabilistic topic distribution, which K-Means then partitions into thematic clusters corresponding to doctrinal groupings within each ESG pillar [2]. These cluster-level representations inform the feature engineering choices carried forward into the predictive phase. The corpus taxonomy is defined exclusively from external regulatory and statutory authority — the GHG Protocol Scopes 1–3 for Environmental conduct, the UN Guiding Principles on Business and Human Rights (2011) [3] and Uyghur Forced Labor Prevention Act (2021) [4] for Social, and ERISA § 404 fiduciary duties [5] and DOL Final ESG Rule 29 CFR § 2550.404a-1 (2022) [6] for Governance — ensuring that label construction is corpus-agnostic and immune to circular reasoning.

A token audit of the 444 filtered cases revealed that 99.1% of documents exceed the 512-token limit imposed by standard BERT-family architectures, with a corpus mean of 8,824 tokens per case [7]. This empirical finding renders DistilBERT and RoBERTa architecturally unworkable for this litigation corpus without severe truncation that would eliminate the operative legal reasoning contained in the latter portions of judicial opinions. The transition to `allenai/longformer-base-4096` [8] is therefore a data-driven architectural necessity: Longformer's sliding-window attention mechanism with global attention on the [CLS] classification token accommodates the full distributional range of judicial opinion length without information loss. To mitigate catastrophic forgetting on a research-scale dataset of 444 cases, the bottom eight encoder layers are frozen during fine-tuning, constraining gradient updates to the upper representational layers where task-specific legal reasoning is most salient [9]. Class-weighted cross-entropy loss penalizes prediction errors on the underrepresented Social pillar — 47 cases, 10.6% of the corpus — and the single confirmed greenwashing case, preserving meaningful gradient signal for sparse but legally significant categories [10].

---

## Slide: Scope and Limitations

The corpus is necessarily circumscribed by the mechanics of Westlaw's ESG litigation search interface, which indexes cases meeting Westlaw's proprietary ESG relevance criteria rather than all extant ESG-adjacent federal and state litigation [11]. This retrieval constraint introduces an unknown selection bias whose direction cannot be determined without an exhaustive independent audit of unpublished decisions and arbitral awards that fall outside Westlaw's indexing perimeter. The resulting 444-case filtered corpus, while sufficiently large to train generalizable transformer representations, remains below the threshold at which deep learning architectures consistently outperform classical ML shallow models — a structural limitation that motivates the RF/XGBoost baseline as a necessary comparator and not merely a pedagogical exercise [12]. Any performance gap between the ML baseline and Longformer should therefore be interpreted as the marginal value of architectural expressiveness on a constrained legal corpus, not as evidence of generalizability to the broader population of unreported ESG litigation.

Two confirmed data integrity issues constrain interpretive reach. First, the noise filter applied in preprocessing incorrectly excluded two greenwashing cases that satisfied the Environmental pillar's evidentiary threshold — reducing the confirmed greenwash count from three expected cases to one — and no post-hoc correction is possible without re-auditing the 474 excluded files against updated keyword criteria [13]. The single surviving greenwash label renders any greenwash-specific evaluation metric statistically non-interpretable and must be disclosed as a recognized limitation of the study's class distribution. Second, the OCR provenance of the Westlaw markdown exports introduces character-level noise in low-quality scanned documents that cannot be fully remediated by the cleaning pipeline, creating the possibility of systematic token-level degradation in older judicial opinions [14]. Hallucination risk in the Longformer's attention mechanism — whereby the model assigns high attribution scores to legally irrelevant co-occurrence patterns rather than substantive doctrinal triggers — is addressed but not eliminated by the SHAP PartitionExplainer and attention heatmap analyses; validation against known legal authority remains a human-review requirement rather than an automated quality gate [15]. Reproducibility is constrained by the stochastic elements of LDA (topic-word sampling), K-Means initialization, and Longformer fine-tuning; all random seeds are fixed and reported, but hardware-level non-determinism in CUDA floating-point operations may produce marginal metric variation across GPU environments [16].

---

## References

[1] Blei, D.M., Ng, A.Y., & Jordan, M.I. (2003). Latent Dirichlet Allocation. *Journal of Machine Learning Research*, 3, 993–1022.
[2] MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the 5th Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281–297.
[3] UN Office of the High Commissioner for Human Rights. (2011). *Guiding Principles on Business and Human Rights*. United Nations.
[4] Uyghur Forced Labor Prevention Act, Pub. L. No. 117-78, 135 Stat. 1525 (2021).
[5] Employee Retirement Income Security Act of 1974, 29 U.S.C. § 1104 (§ 404 fiduciary duties of prudence and loyalty).
[6] U.S. Dep't of Labor. (2022). Prudence and Loyalty in Selecting Plan Investments and Exercising Shareholder Rights. 29 C.F.R. § 2550.404a-1.
[7] Internal token audit: `05_esg_text_clean.py`, April 22, 2026. Mean 8,824 tokens; 99.1% (440/444) exceed 512.
[8] Beltagy, I., Peters, M.E., & Cohan, A. (2020). Longformer: The Long-Document Transformer. *arXiv:2004.05150*.
[9] Howard, J., & Ruder, S. (2018). Universal Language Model Fine-Tuning for Text Classification. *ACL 2018*, 328–339.
[10] King, G., Zeng, L. (2001). Logistic Regression in Rare Events Data. *Political Analysis*, 9(2), 137–163.
[11] Westlaw. (2024). *ESG Litigation Search Methodology*. Thomson Reuters.
[12] Kaplan, J. et al. (2020). Scaling Laws for Neural Language Models. *arXiv:2001.08361*.
[13] Internal filter audit: `02_non_esg_filter_noise.py` manifest, April 22, 2026. Greenwash false negatives: 2 cases.
[14] Tesseract OCR. (2023). *Tesseract Documentation*. Google. https://github.com/tesseract-ocr/tesseract.
[15] Lundberg, S.M., & Lee, S-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017*, 4765–4774.
[16] Paszke, A. et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS 2019*, 8024–8035.
