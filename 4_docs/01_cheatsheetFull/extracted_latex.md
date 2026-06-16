# Extracted LaTeX Source Code: Figures and Tables

This document contains the exact LaTeX source code for all the figures and tables defined in the report. Each section includes the relative path and absolute path to the file containing the code, the line numbers, and the raw code block.

---

## 1. Figures (Chapter 3)

### Figure 3.1: Overall architecture of the proposed multi-phase distillation framework
* **Source File:** [figs/tikz/fig_3_1_overall_pipeline.tex](file:///E:/DoCode/1%20VN-Legal-AI/01_Report_CD1_DistillationLaw/figs/tikz/fig_3_1_overall_pipeline.tex)
* **Lines:** 1-30

```latex
% Figure 3.1 - Overall three-phase pipeline
\begin{figure}[H]
\centering
\begin{tikzpicture}[
    scale=0.84,
    transform shape,
    font=\small,
    block/.style={rectangle, draw=blue!50!black, fill=blue!8, text width=2.3cm,
                  align=center, minimum height=1.6cm, rounded corners=3pt},
    final/.style={rectangle, draw=green!50!black, fill=green!10, text width=2.3cm,
                  align=center, minimum height=1.6cm, rounded corners=3pt},
    arrow/.style={-{Stealth[length=2.5mm]}, thick}
]
    \node (sft)    [block] {\textbf{SFT Base}\\ \footnotesize Qwen3\\ \footnotesize 0.6B / 1.7B};
    \node (phase2) [block, above right=0.4cm and 1.1cm of sft] {\textbf{Phase 2}\\ \footnotesize On-Policy KD\\ \footnotesize Oracle + KL Clip};
    \node (phase1) [block, below right=0.4cm and 1.1cm of sft] {\textbf{Phase 1}\\ \footnotesize Offline Logit KD\\ \footnotesize $\mathcal{L}_{CE} + \mathcal{L}_{KD}$};
    \node (phase3) [block, right=1.0cm of phase1] {\textbf{Phase 3}\\ \footnotesize Diagnosis-Driven\\ \footnotesize DPO Alignment};
    \node (out3)   [final, right=0.7cm of phase3] {\textbf{DPO-Aligned}\\ \textbf{SLM}};
    \node (out2)   [final] at (out3 |- phase2) {\textbf{On-Policy}\\ \textbf{SLM}};

    \draw [arrow] (sft.east) -- (phase2.west);
    \draw [arrow] (sft.east) -- (phase1.west);
    \draw [arrow] (phase2.east) -- (out2.west);
    \draw [arrow] (phase1) -- (phase3);
    \draw [arrow] (phase3) -- (out3);
    \draw [arrow, dashed, gray] (phase1.south) to[out=-30,in=-150] node[midway, below, font=\footnotesize, text=gray] {reference policy} (phase3.south);
\end{tikzpicture}
\caption{Overall architecture of the proposed multi-phase distillation framework. From a shared SFT base, two distillation tracks branch \emph{independently}: an on-policy track (Phase 2, on-policy knowledge distillation) and an offline track (Phase 1, offline logit distillation). Phase 3 (diagnosis-driven preference optimization) then refines the Phase 1 checkpoint, which also serves as the DPO reference policy. The two tracks share only the SFT initialization.}
\label{fig:overall_pipeline}
\end{figure}
```

---

### Figure 3.2: Computation flow of offline logit distillation in Phase 1
* **Source File:** [figs/tikz/fig_3_2_offline_logit_kd.tex](file:///E:/DoCode/1%20VN-Legal-AI/01_Report_CD1_DistillationLaw/figs/tikz/fig_3_2_offline_logit_kd.tex)
* **Lines:** 1-43

```latex
% Figure 3.2 - Phase 1: Offline Logit Distillation mechanism
\begin{figure}[H]
\centering
\begin{tikzpicture}[
    font=\small,
    block/.style={rectangle, draw=blue!50!black, fill=blue!8, text width=2.8cm,
                  align=center, minimum height=1.3cm, rounded corners=3pt},
    data/.style={rectangle, draw=gray!70, fill=gray!10, text width=2.1cm,
                 align=center, minimum height=1.3cm, rounded corners=3pt},
    loss/.style={rectangle, draw=red!60!black, fill=red!8, text width=3.3cm,
                 align=center, minimum height=1.3cm, rounded corners=3pt},
    total/.style={rectangle, draw=orange!70!black, fill=orange!12, text width=3.1cm,
                  align=center, minimum height=1.3cm, rounded corners=3pt},
    arrow/.style={-{Stealth[length=2.5mm]}, thick},
    backprop/.style={-{Stealth[length=2.5mm]}, thick, dashed, red!70!black}
]
    \node (input)   [data] {\textbf{Input Data}\\ \footnotesize Prompt + Gold $y^*$};
    \node (teacher) [block, above right=0.35cm and 0.9cm of input] {\textbf{Teacher (Frozen)}\\ \footnotesize Top-50 logits};
    \node (student) [block, below right=0.35cm and 0.9cm of input] {\textbf{Student}\\ \footnotesize QLoRA adapters};
    \node (kdloss)  [loss, right=0.9cm of teacher] {Soft Loss\\ \footnotesize $\mathcal{L}_{KD} = T^2 D_{KL}\!\left(P^{T}_{\text{tea}} \,\|\, P^{T}_{\text{stu}}\right)$};
    \node (celoss)  [loss, right=0.9cm of student] {Hard Loss\\ \footnotesize $\mathcal{L}_{CE}$ (gold tokens)};
    \node (totalloss) [total] at ($($(kdloss.east)!0.5!(celoss.east)$)+(2.6cm,0)$) {\textbf{Total Loss}\\ \footnotesize $\alpha \mathcal{L}_{KD} + (1-\alpha)\mathcal{L}_{CE}$};

    \draw [arrow] (input.north) |- (teacher.west);
    \draw [arrow] (input.south) |- (student.west);
    \draw [arrow] (teacher) -- (kdloss);
    \draw [arrow] (student) -- (celoss);
    \draw [arrow] (student.east) -- ++(0.45,0) |- ($(kdloss.west)+(0,-0.35)$);
    \draw [arrow] (kdloss.east) -- ++(0.35,0) |- ($(totalloss.west)+(0,0.25)$);
    \draw [arrow] (celoss.east) -- ++(0.35,0) |- ($(totalloss.west)+(0,-0.25)$);

    % Backpropagation: route clearly below the lowest box (celoss), label under the line
    \coordinate (bpy) at ($(celoss.south)+(0,-0.9)$);
    \coordinate (bp) at (totalloss.south |- bpy);
    \draw [backprop] (totalloss.south) -- (bp) -| (student.south);
    \node [below=3pt, font=\footnotesize, text=red!70!black]
        at ($(student.south |- bp)!0.5!(bp)$)
        {Backpropagation (update LoRA adapters)};
\end{tikzpicture}
\caption{Computation flow of offline logit distillation in Phase 1. The frozen teacher provides Top-$K$ soft labels, while the trainable student is optimized with a weighted combination of soft-label KL divergence and hard-label cross-entropy anchored on the gold output.}
\label{fig:phase1_distillation}
\end{figure}
```

---

### Figure 3.3: On-policy knowledge distillation loop in Phase 2
* **Source File:** [figs/tikz/fig_3_3_onpolicy_kd.tex](file:///E:/DoCode/1%20VN-Legal-AI/01_Report_CD1_DistillationLaw/figs/tikz/fig_3_3_onpolicy_kd.tex)
* **Lines:** 1-48

```latex
% Figure 3.3 - Phase 2: On-Policy Knowledge Distillation loop
\begin{figure}[H]
\centering
\begin{tikzpicture}[
    font=\small,
    block/.style={rectangle, draw=blue!50!black, fill=blue!8, text width=3cm,
                  align=center, minimum height=1.4cm, rounded corners=3pt},
    data/.style={rectangle, draw=gray!70, fill=gray!10, text width=1.9cm,
                 align=center, minimum height=1.4cm, rounded corners=3pt},
    rollout/.style={rectangle, draw=orange!70!black, fill=yellow!12, text width=3cm,
                    align=center, minimum height=1.4cm, rounded corners=3pt},
    loss/.style={rectangle, draw=red!60!black, fill=red!8, text width=3cm,
                 align=center, minimum height=1.4cm, rounded corners=3pt},
    arrow/.style={-{Stealth[length=2.5mm]}, thick},
    priv/.style={-{Stealth[length=2.5mm]}, thick, dashed, gray},
    backprop/.style={-{Stealth[length=2.5mm]}, thick, dashed, red!70!black}
]
    % Bottom row: generation flow
    \node (prompt)  [data] {\textbf{Prompt} $x$};
    \node (student) [block, right=1.1cm of prompt] {\textbf{Student} $\pi_\theta$\\ \footnotesize stochastic decoding\\ \footnotesize $T{=}0.7$, top-$p{=}0.9$};
    \node (rollout) [rollout, right=1.1cm of student] {\textbf{Rollout} $\hat{y}$\\ \footnotesize on-policy sample\\ \footnotesize $\hat{y} \sim \pi_\theta(\cdot \mid x)$};
    % Top row: supervision flow (klclip vertically above student, teacher above rollout)
    \node (klclip)  [loss, above=1.5cm of student] {\textbf{Clipped KL Loss}\\ \footnotesize $KL_t^{\text{clip}} = \min(KL_t, C)$};
    \node (teacher) [block, above=1.5cm of rollout] {\textbf{Teacher Oracle} $\pi_T$\\ \footnotesize privileged gold $y^*$};

    % Forward flow
    \draw [arrow] (prompt) -- (student);
    \draw [arrow] (student) -- (rollout);
    \draw [arrow] (rollout) -- (teacher)
        node[midway, right, font=\footnotesize] {$\hat{y}$};
    \draw [arrow] (teacher) -- (klclip)
        node[midway, above, font=\footnotesize] {$P^{T}_{\text{tea}}$};
    % Student distribution up to the loss (left lane), update back down (right lane)
    \draw [arrow] ($(student.north)+(-0.6,0)$) -- ($(klclip.south)+(-0.6,0)$)
        node[midway, left, font=\footnotesize] {$P^{T}_{\text{stu}}$};
    \draw [backprop] ($(klclip.south)+(0.6,0)$) -- ($(student.north)+(0.6,0)$)
        node[midway, right, font=\footnotesize] {update $\theta$};
    % Privileged information: routed above the whole diagram, clear of every box
    \coordinate (privy) at ($(klclip.north)+(0,0.55)$);
    \draw [priv] (prompt.north) |- (privy) -| (teacher.north);
    \node [above=3pt, font=\footnotesize, text=gray]
        at ($(klclip.north |- privy)!0.5!(teacher.north |- privy)$)
        {privileged info: $x$, gold $y^*$};
\end{tikzpicture}
\caption{On-policy knowledge distillation loop in Phase 2. The student generates its own rollouts $\hat{y}$; the teacher, operating as an Oracle with privileged access to the gold answer $y^*$, supplies corrective token-level distributions, and the student is updated through clipped KL divergence.}
\label{fig:phase2_onpolicy}
\end{figure}
```

---

### Figure 3.4: Diagnosis-driven preference optimization workflow in Phase 3
* **Source File:** [figs/tikz/fig_3_4_diagnosis_dpo.tex](file:///E:/DoCode/1%20VN-Legal-AI/01_Report_CD1_DistillationLaw/figs/tikz/fig_3_4_diagnosis_dpo.tex)
* **Lines:** 1-36

```latex
% Figure 3.4 - Phase 3: Diagnosis-Driven DPO Alignment workflow
\begin{figure}[H]
\centering
\begin{tikzpicture}[
    font=\small,
    block/.style={rectangle, draw=blue!50!black, fill=blue!8, text width=2.5cm,
                  align=center, minimum height=1.4cm, rounded corners=3pt},
    judge/.style={rectangle, draw=orange!70!black, fill=yellow!12, text width=2.5cm,
                  align=center, minimum height=1.4cm, rounded corners=3pt},
    decision/.style={diamond, draw=orange!70!black, fill=yellow!12, text width=1.5cm,
                     align=center, aspect=1.3, inner sep=1pt},
    ok/.style={rectangle, draw=green!50!black, fill=green!10, text width=2cm,
               align=center, minimum height=0.9cm, rounded corners=3pt},
    dpo/.style={rectangle, draw=red!60!black, fill=red!8, text width=2.5cm,
                align=center, minimum height=1.4cm, rounded corners=3pt},
    arrow/.style={-{Stealth[length=2.5mm]}, thick}
]
    \node (infer)  [block] {\textbf{Student Rollouts}\\ \footnotesize $\hat{y} \sim \pi_\theta(\cdot \mid x)$};
    \node (judge)  [judge, right=0.9cm of infer] {\textbf{LLM Judge}\\ \footnotesize GPT-4o-mini\\ \footnotesize semantic audit};
    \node (decide) [decision, right=0.9cm of judge] {\footnotesize Error?};
    \node (ok)     [ok, above=0.7cm of decide] {\texttt{OK} $\rightarrow$ discard};
    \node (sample) [block, right=0.9cm of decide, fill=gray!10, draw=gray!70] {\textbf{Stratified Sampling}\\ \footnotesize pairs $(x, y_w, y_l)$};
    \node (dpoAlign) [dpo, below=0.8cm of sample] {\textbf{DPO Alignment}\\ \footnotesize $\beta = 0.1$, ref.\ $= \pi_{\text{Phase 1}}$};

    \draw [arrow] (infer) -- (judge);
    \draw [arrow] (judge) -- (decide);
    \draw [arrow] (decide) -- node[left, font=\footnotesize] {No} (ok);
    \draw [arrow] (decide) -- node[above, font=\footnotesize] {Yes} (sample);
    \draw [arrow] (sample) -- (dpoAlign);
    \draw [arrow, dashed, red!70!black] (dpoAlign.west) -| (infer.south)
        node[pos=0.25, below, font=\footnotesize] {align student $\pi_\theta$};
\end{tikzpicture}
\caption{Diagnosis-driven preference optimization workflow in Phase 3. An external LLM Judge audits student rollouts; erroneous samples are paired with gold answers via stratified sampling to form preference pairs $(x, y_w, y_l)$ for DPO training.}
\label{fig:phase3_dpo}
\end{figure}
```

---

## 2. Tables (Chapter 4)

### Table 4.1: Summary of Dataset Splits and Evaluation Sets
* **Source File:** [2_chapters/4_Experiments/4_1_Datasets.tex](file:///E:/DoCode/1%20VN-Legal-AI/01_Report_CD1_DistillationLaw/2_chapters/4_Experiments/4_1_Datasets.tex#L16-L31)
* **Lines:** 16-31

```latex
\begin{table}[htbp]
\centering
\caption{Summary of Dataset Splits and Evaluation Sets}
\label{tab:dataset_summary}
\begin{tabular}{llcccc}
\hline
\textbf{Split} & \textbf{Source / Dataset} & \textbf{MCQ} & \textbf{NLI} & \textbf{Syllogism} & \textbf{Total} \\ \hline
Train & \texttt{train} & 796 & 1,128 & 679 & 2,603 \\
Valid & \texttt{val} & 101 & 141 & 80 & 322 \\
Test (Split) & \texttt{test} & 100 & 141 & 84 & 325 \\
Test (Public) & \href{https://huggingface.co/datasets/VLSP2025-LegalSML/Public-Test}{VLSP2025 Public-Test} & 146 & 150 & 144 & 440 \\ \hline
\end{tabular}

\vspace{2pt}
{\footnotesize \textit{Note}: After cleaning malformed and inconsistently formatted samples, the validation split contains 322 examples, rather than the 343 ($10\%$ of 3,428) obtained from the initial stratified partition.}
\end{table}
```

---

### Table 4.2: Impact of Hyperparameter Tuning (V1 vs. V2) on the 325-sample Split-Test
* **Source File:** [2_chapters/4_Experiments/4_3_Results_and_Analysis.tex](file:///E:/DoCode/1%20VN-Legal-AI/01_Report_CD1_DistillationLaw/2_chapters/4_Experiments/4_3_Results_and_Analysis.tex#L11-L27)
* **Lines:** 11-27

```latex
\begin{table}[htbp]
\centering
\caption{Impact of Hyperparameter Tuning (V1 vs. V2) on the 325-sample Split-Test}
\label{tab:hp_tuning}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccccccccc}
\hline
\textbf{Run} & \textbf{Method} & \textbf{Size} & \textbf{LR} & \textbf{$\alpha$} & \textbf{$T$} & \textbf{MC Acc} & \textbf{NLI Acc} & \textbf{NLI F1} & \textbf{Syl R-L} & \textbf{Overall} \\ \hline
E1 & LoRA V1  & 0.6B & 1.0e-4 & 0.5 & 2.0 & 0.2772 & 0.3475 & 0.3808 & 0.1623 & 0.2560 \\
A1 & LoRA V2  & 0.6B & 2.0e-5 & 0.3 & 1.5 & 0.4059 & 0.2270 & 0.2416 & 0.1741 & \textbf{0.4520} \\
E2 & QLoRA V1 & 0.6B & 1.0e-4 & 0.5 & 2.0 & 0.3564 & 0.4113 & 0.5322 & 0.2104 & 0.2536 \\
A2 & QLoRA V2 & 0.6B & 2.0e-5 & 0.3 & 1.5 & 0.4158 & 0.5886 & 0.4390 & 0.3804 & \textbf{0.4622} \\
E3 & LoRA V1  & 1.7B & 1.0e-4 & 0.5 & 2.0 & 0.5149 & 0.5957 & 0.7025 & 0.3627 & 0.3644 \\
A3 & LoRA V2  & 1.7B & 2.0e-5 & 0.3 & 1.5 & 0.4455 & 0.7305 & 0.5018 & 0.4208 & \textbf{0.5322} \\ \hline
\end{tabular}%
}
\end{table}
```

---

### Table 4.3: Performance Progression Across Training Phases on the 325-sample Split-Test
* **Source File:** [2_chapters/4_Experiments/4_3_Results_and_Analysis.tex](file:///E:/DoCode/1%20VN-Legal-AI/01_Report_CD1_DistillationLaw/2_chapters/4_Experiments/4_3_Results_and_Analysis.tex#L43-L61)
* **Lines:** 43-61

```latex
\begin{table}[htbp]
\centering
\caption{Performance Progression Across Training Phases on the 325-sample Split-Test}
\label{tab:progression}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccc}
\hline
\textbf{Run} & \textbf{Stage} & \textbf{Size} & \textbf{MC Acc} & \textbf{NLI Acc} & \textbf{NLI F1} & \textbf{Syl R-L} & \textbf{Overall} \\ \hline
B2 & SFT (baseline)   & 0.6B & 0.1400 & 0.7160 & 0.4730 & 0.4280 & 0.4282 \\
A2 & Offline KD (V2)  & 0.6B & 0.4158 & 0.5886 & 0.4390 & 0.3804 & 0.4622 \\
A2 & On-Policy KD     & 0.6B & 0.1600 & 0.4960 & 0.3800 & 0.2750 & 0.3104 \\
A2 & DPO Alignment    & 0.6B & 0.1000 & 0.3191 & 0.2387 & 0.1392 & 0.1861 \\ \hline
B3 & SFT (baseline)   & 1.7B & 0.5000 & 0.7450 & 0.5040 & 0.2070 & 0.4840 \\
A3 & Offline KD (V2)  & 1.7B & 0.4455 & 0.7305 & 0.5018 & 0.4208 & 0.5322 \\
A3 & On-Policy KD     & 1.7B & 0.3600 & 0.7870 & 0.5310 & 0.3750 & 0.5075 \\
A3 & DPO Alignment    & 1.7B & 0.2000 & 0.8156 & 0.8333 & 0.5684 & \textbf{0.5280} \\ \hline
\end{tabular}%
}
\end{table}
```

---

### Table 4.4: Generalization Results on the 440-sample VLSP2025 Public-Test (Out-of-Distribution)
* **Source File:** [2_chapters/4_Experiments/4_3_Results_and_Analysis.tex](file:///E:/DoCode/1%20VN-Legal-AI/01_Report_CD1_DistillationLaw/2_chapters/4_Experiments/4_3_Results_and_Analysis.tex#L79-L97)
* **Lines:** 79-97

```latex
\begin{table}[htbp]
\centering
\caption{Generalization Results on the 440-sample VLSP2025 Public-Test (Out-of-Distribution)}
\label{tab:vlsp_ood}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccccccc}
\hline
\textbf{Run} & \textbf{Stage} & \textbf{Size} & \textbf{MC Acc} & \textbf{NLI Acc} & \textbf{NLI F1} & \textbf{Syl R-1} & \textbf{Syl R-L} & \textbf{Overall} \\ \hline
B2 & SFT          & 0.6B & 0.541 & 0.707 & 0.480 & 0.449 & 0.266 & 0.5047 \\
B3 & SFT          & 1.7B & \textbf{0.740} & 0.873 & 0.590 & 0.581 & 0.305 & \textbf{0.6395} \\
A2 & KD V2        & 0.6B & 0.199 & 0.607 & 0.428 & 0.610 & 0.331 & 0.3787 \\
A3 & KD V2        & 1.7B & 0.219 & 0.720 & 0.525 & 0.667 & 0.353 & 0.4307 \\
A2 & KD + DPO     & 0.6B & 0.151 & 0.207 & 0.177 & 0.165 & 0.122 & 0.1599 \\
A3 & KD + DPO     & 1.7B & 0.212 & 0.893 & \textbf{0.897} & 0.409 & 0.282 & 0.4627 \\
A2 & On-Policy KD & 0.6B & 0.055 & 0.673 & 0.482 & 0.609 & 0.313 & 0.3471 \\
A3 & On-Policy KD & 1.7B & 0.555 & \textbf{0.920} & 0.617 & \textbf{0.679} & \textbf{0.372} & 0.6155 \\ \hline
\end{tabular}%
}
\end{table}
```

---

### Table 4.5: Cross-Entropy Loss Trend: LoRA vs. QLoRA on the 0.6B Student (V2 Configuration)
* **Source File:** [2_chapters/4_Experiments/4_4_Ablation_Study.tex](file:///E:/DoCode/1%20VN-Legal-AI/01_Report_CD1_DistillationLaw/2_chapters/4_Experiments/4_4_Ablation_Study.tex#L11-L23)
* **Lines:** 11-23

```latex
\begin{table}[htbp]
\centering
\caption{Cross-Entropy Loss: LoRA vs. QLoRA on the 0.6B Student (V2 Configuration)}
\label{tab:qlora_ablation}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccc}
\hline
\textbf{Adapter} & \textbf{CE Loss ($L_{CE}$: start $\rightarrow$ settle)} & \textbf{Knowledge Retention} & \textbf{Syllogism R-L} \\ \hline
LoRA (A1)  & $1.00 \rightarrow 0.56$ (stays high)        & Higher CE (weaker)   & 0.1741 \\
QLoRA (A2) & $0.64 \rightarrow 0.37$ (steady $\downarrow$) & Lower CE (anchored)  & \textbf{0.3804} \\ \hline
\end{tabular}%
}
\end{table}
```

---

### Table 4.6: Diagnosis Quality on the 0.6B Student Training Rollouts: Rule-Based vs. LLM Judge
* **Source File:** [2_chapters/4_Experiments/4_4_Ablation_Study.tex](file:///E:/DoCode/1%20VN-Legal-AI/01_Report_CD1_DistillationLaw/2_chapters/4_Experiments/4_4_Ablation_Study.tex#L31-L46)
* **Lines:** 31-46

```latex
\begin{table}[htbp]
\centering
\caption{Diagnosis Quality on the 0.6B Student Training Rollouts: Rule-Based vs. LLM Judge}
\label{tab:judge_ablation}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccc}
\hline
\textbf{Diagnoser} & \textbf{OK (acceptable)} & \textbf{RISKY (lucky guess)} & \textbf{WRONG (detected)} \\ \hline
Rule-Based (Regex + ROUGE-L) & 290 (11.1\%) & 898 (34.5\%) & 1{,}054 (40.5\%) \\
LLM Judge (\texttt{GPT-4o-mini}) & \textbf{538 (20.7\%)} & 404 (15.5\%) & \textbf{1{,}652 (63.5\%)} \\ \hline
\end{tabular}%
}

\vspace{2pt}
{\footnotesize \textit{Note}: Percentages are computed over all 2{,}603 training rollouts; the residual \texttt{PARTIAL} class is omitted, so the three columns do not sum to 100\%.}
\end{table}
```

