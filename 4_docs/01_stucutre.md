# Khung Luận Văn — Legal SLM Distillation

### **Chapter 1 — Introduction**

1.1 Reason for choosing the topic

1.2 Research objectives
- 1.2.1 General objective
- 1.2.2 Specific objectives

1.3 Research subjects and scope
- 1.3.1 Research subjects
- 1.3.2 Research scope

1.4 Contributions of the study

1.5 Report structure

---

### **Chapter 2 — Background and Literature Review**

**2.1 Small Language Models in Legal Domain**
- 2.1.1 Opportunities and Constraints
- 2.1.2 Survey of Legal Adaptation Frameworks
- 2.1.3 Research Gaps in Legal Distillation

**2.2 Knowledge Distillation for Language Models**
- 2.2.1 Classical Off-Policy Logit Distillation
- 2.2.2 Exposure Bias in Sequential Reasoning
- 2.2.3 On-Policy Distillation
- 2.2.4 Top-K Sparse Logits vs. Full-Vocabulary Distillation

**2.3 Parameter-Efficient Fine-Tuning**
- 2.3.1 Low-Rank Adaptation
- 2.3.2 Quantized Low-Rank Adaptation as Implicit Regularization

**2.4 Reinforcement Learning for Legal Reasoning**
- 2.4.1 Group Relative Policy Optimization
- 2.4.2 Privileged Information in Distillation

**2.5 Preference Optimization and Alignment**
- 2.5.1 Direct Preference Optimization
- 2.5.2 Large Language Model as a Judge Paradigm

**2.6 Evaluation Metrics in Legal Reasoning**
- 2.6.1 Classification Metrics
- 2.6.2 Generation Metrics

---

### **Chapter 3 — Proposed Methods and Models**

**3.1 Overview of the Proposed Pipeline**
- 3.1.1 Problem Formulation for Legal Reasoning
- 3.1.2 Overall Architecture of the Framework

**3.2 Phase 1: Offline Logit Distillation**
- 3.2.1 Supervised Training with Anchor
- 3.2.2 Soft-Label Training Objective with Top-K Logits
- 3.2.3 Regularization via Quantized Low-Rank Adaptation
- 3.2.4 Hyperparameter Design and Justification

**3.3 Phase 2: On-Policy Knowledge Distillation**
- 3.3.1 Motivation: Breaking Exposure Bias
- 3.3.2 Student-Generated Rollouts
- 3.3.3 Teacher as Oracle with Privileged Information
- 3.3.4 Kullback-Leibler Divergence Clipping
- 3.3.5 Two-Phase Loss Design

**3.4 Phase 3: Diagnosis-Driven Preference Optimization**
- 3.4.1 Exploration and Error Diagnosis
- 3.4.2 Preference Pair Construction
- 3.4.3 Rule-Based vs. Large Language Model Judge
- 3.4.4 Preference Alignment via Direct Preference Optimization

---

### **Chapter 4 — Experiments**

**4.1 Datasets**
- 4.1.1 Training Datasets
- 4.1.2 Evaluation and Test Sets
- 4.1.3 Direct Preference Optimization Preference Data

**4.2 Experimental Setup**
- 4.2.1 Teacher and Student Models
- 4.2.2 Configuration Space and Hyperparameters
- 4.2.3 Hardware and Framework

**4.3 Results and Analysis**
- 4.3.1 Impact of Hyperparameter Tuning
- 4.3.2 Low-Rank Adaptation vs. Quantized Low-Rank Adaptation
- 4.3.3 Model Size Comparison: 0.6 Billion vs. 1.7 Billion Parameters
- 4.3.4 Supervised Fine-Tuning vs. Offline Knowledge Distillation
- 4.3.5 Offline vs. On-Policy Knowledge Distillation
- 4.3.6 Direct Preference Optimization Alignment Impact
- 4.3.7 Generalization on the Public Test Set

**4.4 Ablation Study**
- 4.4.1 Quantized Low-Rank Adaptation vs. Low-Rank Adaptation
- 4.4.2 Rule-Based vs. Large Language Model Judge

**4.5 Qualitative Analysis**
- 4.5.1 Nine-Sample Inference Comparison
- 4.5.2 Error Patterns Before and After Distillation

---

### **Chapter 5 — Conclusion and Future Work**

5.1 Conclusion

5.2 Limitations

5.3 Future work
