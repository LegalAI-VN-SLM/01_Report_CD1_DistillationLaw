# Explanation of Figures and Tables: Multi-Phase Distillation for Vietnamese Legal QA

This document provides a comprehensive overview and analysis of the figures and tables defined in the report on Vietnamese Legal Question Answering Distillation. It explains the theoretical background, implementation details, experimental findings, and insights represented by each figure and table.

---

## 1. Architectural and Methodological Figures (Chapter 3)

These figures illustrate the overall multi-phase distillation framework and the detailed computational flow of each phase.

### Figure 3.1: Overall Architecture of the Distillation Framework
* **Source File:** [fig_3_1_overall_pipeline.tex](file:///E:/DoCode/1%20VN-Legal-AI/01_Report_CD1_DistillationLaw/figs/tikz/fig_3_1_overall_pipeline.tex)
* **Description:** Represents the high-level routing of our distillation pipeline. 
* **Key Details:**
  * **Input (shared root):** Starts with a Supervised Fine-Tuning (SFT) Base model (Qwen3-0.6B or 1.7B). This SFT base is the only thing the two tracks share.
  * **Two independent tracks (not a linear chain):** From the SFT base, two tracks branch in parallel:
    * **On-policy track — Phase 2 (On-Policy KD):** Applied **directly to the SFT base** (independent of Phase 1). Addresses exposure bias using a teacher Oracle and clipped KL divergence.
    * **Offline track — Phase 1 (Offline Logit KD) → Phase 3 (Diagnosis-Driven DPO):** Phase 1 trains from the SFT base on static teacher logits; **Phase 3 then continues from the Phase 1 checkpoint**, optimizing the student using semantic error-diagnosis pairs $(x, y_w, y_l)$ from an LLM Judge. The Phase 1 checkpoint is also the DPO reference policy.
  * **Outputs:** Produces two distinct final variants: **On-Policy SLM** (top track) and **DPO-Aligned SLM** (bottom track).

### Figure 3.2: Computation Flow of Offline Logit Distillation (Phase 1)
* **Source File:** [fig_3_2_offline_logit_kd.tex](file:///E:/DoCode/1%20VN-Legal-AI/01_Report_CD1_DistillationLaw/figs/tikz/fig_3_2_offline_logit_kd.tex)
* **Description:** Shows how the student is trained to mimic the frozen teacher's probability distribution over a static dataset.
* **Key Details:**
  * The input prompt and gold output $y^*$ are passed to both the frozen **Teacher** and the trainable **Student** (equipped with QLoRA adapters).
  * The **Soft Loss ($\mathcal{L}_{KD}$)** computes the KL divergence between the teacher's Top-50 soft logits ($P^T_{\text{tea}}$) and the student's probability distribution ($P^T_{\text{stu}}$) at temperature $T$.
  * The **Hard Loss ($\mathcal{L}_{CE}$)** is standard cross-entropy anchored directly on the gold tokens.
  * The **Total Loss** is a weighted combination: $\alpha \mathcal{L}_{KD} + (1-\alpha)\mathcal{L}_{CE}$.
  * Backpropagation updates only the student's QLoRA adapters.

### Figure 3.3: On-Policy Knowledge Distillation Loop (Phase 2)
* **Source File:** [fig_3_3_onpolicy_kd.tex](file:///E:/DoCode/1%20VN-Legal-AI/01_Report_CD1_DistillationLaw/figs/tikz/fig_3_3_onpolicy_kd.tex)
* **Description:** Illustrates the closed-loop system designed to bridge the training-inference alignment gap (exposure bias).
* **Key Details:**
  * The student model ($\pi_\theta$) stochastically decodes rollouts $\hat{y}$ (using temperature $T=0.7$ and nucleus sampling $p=0.9$) from the input prompt $x$.
  * The rollout $\hat{y}$ is passed to the **Teacher Oracle ($\pi_T$)**, which has privileged access to the gold answer $y^*$. The teacher supplies corrective token-level distributions ($P^T_{\text{tea}}$) conditioning on this self-generated sequence.
  * The student is updated via **Clipped KL Loss** ($KL_t^{\text{clip}} = \min(KL_t, C)$) to prevent gradient explosion on highly divergent paths.
  * **Privileged Information path:** Shows how the teacher oracle utilizes $x$ and the gold $y^*$ to guide the student's on-policy correction.

### Figure 3.4: Diagnosis-Driven DPO Workflow (Phase 3)
* **Source File:** [fig_3_4_diagnosis_dpo.tex](file:///E:/DoCode/1%20VN-Legal-AI/01_Report_CD1_DistillationLaw/figs/tikz/fig_3_4_diagnosis_dpo.tex)
* **Description:** Details how the student is aligned using pairs of correct answers and its own diagnosed mistakes.
* **Key Details:**
  * The student generates rollouts ($\hat{y} \sim \pi_\theta$).
  * An external **LLM Judge** (`GPT-4o-mini`) audits the rollouts semantically.
  * If no error is found (OK), the rollout is discarded.
  * If an error is detected, the sample is sent to **Stratified Sampling** to form a preference pair $(x, y_w, y_l)$, where $y_w$ is the chosen (gold) response and $y_l$ is the rejected (erroneous) student rollout.
  * The preference pair is used in **DPO Alignment** with $\beta = 0.1$ and the **Phase 1 (offline KD) checkpoint** as both the initialization and the reference model, updating the student $\pi_\theta$.

---

## 2. Experimental and Results Tables (Chapter 4)

These tables document the datasets, configuration tuning, training progressions, generalizations, ablation studies, and qualitative evaluations.

### Table 4.1: Summary of Dataset Splits and Evaluation Sets
* **Source File:** [4_1_Datasets.tex](file:///E:/DoCode/1%20VN-Legal-AI/01_Report_CD1_DistillationLaw/2_chapters/4_Experiments/4_1_Datasets.tex#L16-L31)
* **Description:** Summarizes the size and composition of the training, validation, and test sets.
* **Key Insights:**
  * The dataset is split into **Train** (2,603 samples), **Valid** (322 samples after cleaning), and **Test (Split)** (325 samples) representing the in-distribution partitions from the ViLawQA corpus.
  * The external **VLSP2025 Public-Test** (440 samples) acts as the out-of-distribution (OOD) benchmark.
  * Tasks are categorized into **MCQ** (Multiple Choice QA), **NLI** (Natural Language Inference), and **Syllogism** (deductive reasoning).

### Table 4.2: Impact of Hyperparameter Tuning (V1 vs. V2) on the Split-Test
* **Source File:** [4_3_Results_and_Analysis.tex](file:///E:/DoCode/1%20VN-Legal-AI/01_Report_CD1_DistillationLaw/2_chapters/4_Experiments/4_3_Results_and_Analysis.tex#L11-L27)
* **Description:** Compares the initial configuration (V1) with the corrected configuration (V2) across different models and adaptation methods.
* **Key Insights:**
  * **V1 configuration** (LR $1.0 \times 10^{-4}$, weight $\alpha=0.5$, temp $T=2.0$) led to **catastrophic forgetting**, where the student discarded its prior SFT knowledge. This caused the model to produce incoherent, code-switched generations.
  * **V2 configuration** (LR $2.0 \times 10^{-5}$, weight $\alpha=0.3$, temp $T=1.5$, larger batch size) stabilized training and allowed smooth cross-entropy convergence.
  * The tuning resulted in substantial improvements in the Overall score across all runs, especially for the 1.7B model (run E3 $\rightarrow$ A3), which improved from **0.3644** to **0.5322**.

### Table 4.3: Performance Progression Across Training Phases on the Split-Test
* **Source File:** [4_3_Results_and_Analysis.tex](file:///E:/DoCode/1%20VN-Legal-AI/01_Report_CD1_DistillationLaw/2_chapters/4_Experiments/4_3_Results_and_Analysis.tex#L43-L61)
* **Description:** Tracks how performance changes as the models go through SFT, Offline KD, On-Policy KD, and DPO alignment.
* **Key Insights:**
  * **0.6B Model Capacity Constraint:** On-policy distillation and DPO were highly detrimental to the 0.6B student, causing a logical collapse (Overall score dropping from 0.4622 to 0.3104, and then to 0.1861). The model scale was too small to absorb these alignment objectives without destabilizing its fragile reasoning behaviors.
  * **1.7B Model Refinement:** The 1.7B student successfully benefited from alignment. DPO alignment (A3) pushed NLI Accuracy to **0.8156** and Syllogism ROUGE-L to **0.5684** (the highest in-distribution scores for these tasks), though MCQ accuracy dropped to **0.2000**.
  * **Complementary Strengths:** Distillation acts as an error-corrector. SFT baselines are often strong in one area but weak in others; KD successfully transfers the teacher's balanced capabilities to correct specific gaps.

### Table 4.4: Generalization Results on the VLSP2025 Public-Test (Out-of-Distribution)
* **Source File:** [4_3_Results_and_Analysis.tex](file:///E:/DoCode/1%20VN-Legal-AI/01_Report_CD1_DistillationLaw/2_chapters/4_Experiments/4_3_Results_and_Analysis.tex#L79-L97)
* **Description:** Reports the generalization capabilities of all models on the out-of-distribution external benchmark.
* **Key Insights:**
  * **SFT Baseline Generalization:** The 1.7B SFT baseline (B3) achieved the highest overall score (0.6395), indicating that offline distillation can sometimes over-specialize a student to the training distribution.
  * **On-Policy KD Generalization:** Among the distilled models, the 1.7B On-Policy KD model (A3) was the strongest generalizer (Overall 0.6155), achieving the highest NLI Accuracy (0.920) and Syllogism ROUGE-L (0.372) in the entire evaluation table. This proves that addressing exposure bias significantly boosts generalization on unseen distributions.
  * **MCQ Weakness:** MCQ accuracy remained a persistent challenge for all distilled models compared to the SFT baselines.

### Table 4.5: Cross-Entropy Loss Trend: LoRA vs. QLoRA (0.6B Student, V2 Configuration)
* **Source File:** [4_4_Ablation_Study.tex](file:///E:/DoCode/1%20VN-Legal-AI/01_Report_CD1_DistillationLaw/2_chapters/4_Experiments/4_4_Ablation_Study.tex#L11-L23)
* **Description:** Ablates the adapter choice by comparing standard LoRA with 4-bit QLoRA.
* **Key Insights:**
  * **LoRA (A1):** The cross-entropy loss stays elevated, settling around 0.56 after the warmup peak (and drifting slightly upward late), indicating that standard low-rank adaptation retains SFT knowledge less well. The resulting Syllogism ROUGE-L was **0.1741**.
  * **QLoRA (A2):** Freezing base weights in 4-bit NF4 representation served as an **implicit regularizer**. The CE loss decreases steadily to around 0.37, staying consistently below the LoRA curve, and Syllogism ROUGE-L surged to **0.3804**.

### Table 4.6: Diagnosis Quality on the 0.6B Rollouts: Rule-Based vs. LLM Judge
* **Source File:** [4_4_Ablation_Study.tex](file:///E:/DoCode/1%20VN-Legal-AI/01_Report_CD1_DistillationLaw/2_chapters/4_Experiments/4_4_Ablation_Study.tex#L31-L46)
* **Description:** Compares rule-based (Regex + ROUGE-L) error detection with semantic LLM Judge detection on student rollouts.
* **Key Insights:**
  * **Rule-Based Diagnoser:** Suffered from severe false positives and negatives, recognizing only 11.1% of valid responses as OK due to formatting deviations, and classifying 34.5% of flawed responses with correct answers (lucky guesses) as acceptable (Risky).
  * **LLM Judge:** Correctly audited the reasoning chain, raising the OK count to 20.7% (by accepting format-deviant but correct answers) and demoting lucky guesses, which expanded the identified WRONG samples to 63.5%. This yielded a much cleaner dataset for DPO.


