# Khung Luận Văn — Legal SLM Distillation

Khung luận văn đã được tinh chỉnh để phản ánh đúng pipeline 3 giai đoạn trong mã nguồn:
**Offline KD → On-Policy KD → DPO**, với On-Policy KD từ GRPO Teacher là đóng góp chính.

---

### **Chapter 1 — Introduction**

1.1 Reason for choosing the topic *(Tầm quan trọng của Legal AI, giới hạn về bảo mật/chi phí của LLMs lớn, và sự cần thiết phải chưng cất tri thức xuống SLMs)*

1.2 Research objectives
- 1.2.1 General objective
- 1.2.2 Specific objectives

1.3 Research subjects and scope
- 1.3.1 Research subjects *(Các mô hình ngôn ngữ nhỏ - SLMs, dữ liệu pháp lý: NLI, Syllogism, Multi-choice)*
- 1.3.2 Research scope

1.4 Contributions of the study *(Đóng góp chính: On-Policy KD pipeline phá vỡ Exposure Bias cho suy luận pháp lý)*

1.5 Report structure

---

### **Chapter 2 — Background and Literature Review**

**2.1 Small Language Models in Legal Domain**
- 2.1.1 Opportunities and constraints *(Vấn đề bảo mật dữ liệu khách hàng, triển khai cục bộ on-device)*
- 2.1.2 Reasoning challenges in Legal Artificial Intelligence *(Khó khăn của SLMs: nhảy cóc logic, thiên kiến từ vựng, hiểu sai điều luật)*

**2.2 Knowledge Distillation for Language Models**
- 2.2.1 Classical Off-Policy Logit Distillation *(Chưng cất truyền thống: Teacher sinh logits trên gold answer, Student học theo)*
- 2.2.2 Exposure Bias in Sequential Reasoning *(Lệch phân phối khi Student tự sinh — chỉ thấy gold khi train, phải tự xoay sở khi inference)*
- 2.2.3 On-Policy Distillation *(Giải pháp: Student tự sinh → Teacher chấm trên output của Student — phá vỡ Exposure Bias. Tham khảo: GKD, OPSD)*
- 2.2.4 Top-K Sparse Logits vs Full-Vocabulary Distillation *(Chưng cất chọn lọc top-k xác suất để tối ưu bộ nhớ, so với full-vocab cho On-Policy)*

**2.3 Parameter-Efficient Fine-Tuning**
- 2.3.1 Low-Rank Adaptation *(Low-Rank Adaptation: thêm ma trận rank thấp vào attention layers)*
- 2.3.2 Quantized Low-Rank Adaptation as implicit regularization *(Đóng băng base weights ở 4-bit — tự nhiên chống catastrophic forgetting)*

**2.4 Reinforcement Learning for Legal Reasoning**
- 2.4.1 Group Relative Policy Optimization *(Cơ chế huấn luyện Teacher model cho suy luận Tam đoạn luận)*
- 2.4.2 Privileged Information in Distillation *(Cho Teacher xem gold answer trước khi chấm → Oracle mode, logits chính xác hơn)*

**2.5 Preference Optimization and Alignment**
- 2.5.1 Direct Preference Optimization *(Căn chỉnh trực tiếp trên cặp dữ liệu ưu tiên, không cần reward model)*
- 2.5.2 Large Language Model as a Judge Paradigm *(Ứng dụng mô hình lớn (GPT-4o-mini) làm giám khảo đánh giá ngữ nghĩa thay cho Regex)*

**2.6 Evaluation Metrics in Legal Artificial Intelligence**
- 2.6.1 Classification Metrics *(Accuracy, F1-Score cho NLI và Multi-choice)*
- 2.6.2 Generation Metrics *(ROUGE-1, ROUGE-L cho Syllogism)*

---

### **Chapter 3 — Proposed Methods and Models**

**3.1 Overview of the Proposed Pipeline**
- 3.1.1 Problem formulation for Legal Natural Language Inference, Syllogism, and Multiple-Choice Questions
- 3.1.2 Overall architecture: 3-phase distillation and alignment framework

```
SFT Checkpoint
    │
    ▼
Phase 1: Offline Logit KD (V2)
    │   Teacher forward trên gold answer → Top-K logits
    │   Student: CE loss + KD loss
    ▼
Phase 2: On-Policy KD from GRPO Teacher  ← Main Contribution
    │   Student tự generate → Teacher chấm (Oracle + PI)
    │   KL loss với clipping + CE anchor
    ▼
Phase 3: Diagnosis-Driven DPO
    │   Student inference → LLM Judge chẩn đoán → Preference pairs
    │   DPO alignment
    ▼
Final Model
```

**3.2 Phase 1: Offline Logit Distillation**
- 3.2.1 Supervised training with Supervised Fine-Tuning anchor *(Bảo toàn tri thức cũ bằng CE loss trên nhãn cứng)*
- 3.2.2 Soft-label training objective with Top-K Logits *(KL Divergence loss trên top-50 logits của Teacher)*
- 3.2.3 Regularization using Quantized Low-Rank Adaptation *(Đóng băng base weights 4-bit chống catastrophic forgetting)*
- 3.2.4 Version 1 to Version 2: Hyperparameter corrections *(LR 1e-4→2e-5, alpha 0.5→0.3, batch 16→64 — fix catastrophic forgetting)*

**3.3 Phase 2: On-Policy Knowledge Distillation from Group Relative Policy Optimization Teacher**
- 3.3.1 Motivation: Breaking Exposure Bias *(Offline KD: Student chỉ thấy gold → trượt khi tự generate. On-Policy: Student học từ sai lầm của chính mình)*
- 3.3.2 Student-generated rollouts *(Student tự sinh response với sampling (T=0.7, top_p=0.9) → đa dạng hóa quỹ đạo suy luận)*
- 3.3.3 Teacher as Oracle with Privileged Information *(Inject gold answer vào teacher prompt → Teacher chấm với độ tự tin cực cao)*
- 3.3.4 Per-token Kullback-Leibler Clipping *(Ngăn XML structural tokens (`<think>`, `<major_premise>`) chiếm hết gradient — clip tại ngưỡng C)*
- 3.3.5 Two-phase loss design *(CE trên gold anchor + KD trên student-generated, alpha >= 0.5 ưu tiên On-Policy)*

**3.4 Phase 3: Diagnosis-Driven Preference Optimization**
- 3.4.1 Exploration and Error Diagnosis *(Sinh chuỗi suy luận bằng Student, chẩn đoán bằng GPT-4o-mini: OK/WRONG/RISKY/PARTIAL)*
- 3.4.2 Preference Pair Construction *(Chosen = Gold Truth, Rejected = Student answer bị chẩn đoán sai)*
- 3.4.3 Rule-based vs Large Language Model Judge filtering *(So sánh 2 phương pháp tạo preference pairs)*
- 3.4.4 Preference Alignment via Direct Preference Optimization *(Tối ưu hóa đối kháng giúp SLM tránh các lối suy luận sai)*

---

### **Chapter 4 — Experiments**

**4.1 Datasets**
- 4.1.1 Training datasets *(Nguồn gốc, cách xây dựng, thống kê)*
- 4.1.2 Evaluation and test sets
- 4.1.3 Direct Preference Optimization preference data: Rule-based (2313/2283 pairs) và LLM Judge (2065/1434 pairs)

**4.2 Experimental Setup**
- 4.2.1 Teacher and Student Models *(Teacher: Qwen3-4B-legal-GRPO (8-bit), Students: Qwen3-0.6B (QLoRA) và 1.7B (LoRA))*
- 4.2.2 Configuration space and Hyperparameters *(V2: LR=2e-5, alpha=0.3 cho Offline KD; alpha=0.5, kl_clip=5.0 cho On-Policy)*
- 4.2.3 Hardware and framework *(Kaggle T4 16GB, Unsloth, TRL, bitsandbytes)*
- 4.2.4 Bug fixes and lessons learned *(3 bugs: /think prefix, do_sample=True, max_new_tokens=256)*

**4.3 Results and Analysis**
- 4.3.1 Version 1 vs Version 2: Impact of hyperparameter tuning *(Chứng minh LR quá cao gây catastrophic forgetting)*
- 4.3.2 Comparison of Low-Rank Adaptation and Quantized Low-Rank Adaptation at 0.6 Billion Parameters *(QLoRA thắng nhờ implicit regularization)*
- 4.3.3 Model size comparison: 0.6 Billion vs 1.7 Billion Parameters
- 4.3.4 Supervised Fine-Tuning vs Offline Knowledge Distillation: Effect of knowledge distillation
- 4.3.5 Offline Knowledge Distillation vs On-Policy Knowledge Distillation: Breaking Exposure Bias *(So sánh chính — Main Contribution)*
- 4.3.6 Impact of Direct Preference Optimization alignment *(KD → KD+DPO cải thiện về logic suy luận)*

**4.4 Ablation Study**
- 4.4.1 Quantized Low-Rank Adaptation vs Low-Rank Adaptation *(CE loss trend chứng minh QLoRA ngăn forgetting)*
- 4.4.2 Diagnosis quality: Rule-based vs Large Language Model Judge *(LLM Judge chính xác hơn: A3 OK 44.9% vs 12.3%)*

**4.5 Qualitative Analysis**
- 4.5.1 Nine-sample inference comparison *(Phân tích chi tiết output A1/A2/A3 trên NLI, MC, Syllogism)*
- 4.5.2 Error patterns before and after distillation *(Ví dụ cụ thể: đa ngôn ngữ → tiếng Việt, sai logic → đúng)*

---

### **Chapter 5 — Conclusion and Future Work**

5.1 Conclusion *(Tóm tắt pipeline 3 phases và kết quả chính)*

5.2 Limitations
- Chi phí gọi API cho LLM Judge (GPT-4o-mini)
- On-Policy KD chậm hơn Offline KD (generation overhead)
- Syllogism vẫn là task khó nhất (thiếu training data chuyên biệt)
- Hardware constraints: A3 On-Policy cần GPU lớn hơn T4

5.3 Future work
- Full-vocabulary distillation (bỏ Top-K khi có GPU đủ lớn)
- Teacher BF16 thay vì 8-bit (giữ dark knowledge tốt hơn)
- Multi-task On-Policy KD (không chỉ Syllogism mà cả NLI, MC)
- Self-play iterative distillation (Student → Teacher → Student)
