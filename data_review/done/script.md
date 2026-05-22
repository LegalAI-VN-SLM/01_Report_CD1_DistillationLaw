# Script Thuyết Trình — Legal SLM Full Pipeline Report

---

## Section 1: The Origin — 3,428 Legal Questions

> Chào mọi người. Hôm nay mình sẽ trình bày toàn bộ pipeline xây dựng **Legal SLM** — từ dữ liệu thô đến model pháp lý hoàn chỉnh.
>
> Mọi thứ bắt đầu từ bộ dữ liệu **ViLawQA** trong cuộc thi VLSP 2023: **3,428 câu hỏi pháp luật** chia thành 3 dạng — Multi-choice (1,007), NLI (1,421), và Syllogism (1,000).
>
> Vấn đề là dataset chỉ có câu hỏi và đáp án — **thiếu hoàn toàn phần giải thích**. Student model không thể học suy luận nếu không có reasoning. Đó là lý do cần Knowledge Distillation.

---

## Section 2: Distillation Pipeline

> Pipeline chưng cất gồm 4 module: **Loader** tải model, **Generator** chạy inference, **Parser** trích xuất và kiểm tra chất lượng, và **Pipeline** điều phối toàn bộ.
>
> **Nguyên tắc quan trọng nhất:** chia data 80/10/10 **trước** khi distill. Nếu chạy toàn bộ qua Teacher rồi mới chia, Student sẽ bị data leakage — nó sẽ thấy test data trong quá trình train. Chỉ phần train (2,742 records) được đưa qua Teacher.

---

## Section 3: First Experiment — SeaLLMs-v3-7B

> Thử nghiệm đầu tiên với 20 mẫu multi-choice — kết quả rất tệ.
>
> Parse thành công 80% nhưng accuracy chỉ **40%**, answer sạch chỉ **25%**. Reasoning quality 3/10.
>
> Phát hiện hàng loạt vấn đề: format answer hỗn loạn — lúc "C", lúc cả câu dài, lúc "E" dù bài chỉ có 4 đáp án. Reasoning nhiều dòng chỉ copy lại đáp án. Teacher trả sai — reasoning nói B nhưng answer ghi A. Confidence 0.95 là hardcode, không phải confidence thật.

---

## Section 4: Three Root Causes

> Điều quan trọng nhất mình phát hiện: **vấn đề không nằm ở Teacher — nằm ở Data Schema và Prompt Objective**.
>
> **Root Cause #1:** File converter `to_jsonl.py` tạo ra choices bị trùng lặp. 3 trong 4 options NLI đều là "không liên quan" — Student sẽ học shortcut thay vì học semantic relevance. Thêm nữa, `label_semantics` ghi "0=relevant" nhưng choices[0] lại là "không liên quan" — mâu thuẫn.
>
> **Root Cause #2:** SeaLLMs-v3 là model đa ngôn ngữ. Khi prompt không đủ chặt, nó tự chuyển sang tiếng Anh để reasoning. Student sẽ học style song ngữ — hoàn toàn không mong muốn cho legal SLM tiếng Việt.
>
> **Root Cause #3:** Prompt cũ hỏi Teacher "Câu hỏi này có trả lời được không?" — Teacher trả lời "answerable" thay vì trả lời nội dung pháp lý thực sự. Đây là classification, không phải QA.
>
> **Bài học:** `teacher_valid = true` chỉ có nghĩa parser parse được JSON — không có nghĩa sample tốt.

---

## Section 5: Gold-Conditioned Reasoning

> Đây là bước ngoặt. Thay vì để Teacher tự chọn đáp án rồi giải thích, mình **đổi bài toán**: "Đáp án đúng là X, hãy giải thích vì sao."
>
> Chiến lược cũ: Teacher tự chọn answer + reasoning → accuracy chỉ ~40%.
> Chiến lược mới: Teacher chỉ sinh reasoning vì đã biết đáp án đúng → answer luôn đúng, chỉ cần filter reasoning.
>
> Đội hình Teacher: **CMC-AI-Legal-32B** làm Teacher chính kiêm Judge, **SeaLLMs-v3-7B** sinh reasoning khi đã biết gold answer, và **Qwen2.5-7B** làm Teacher phụ.

---

## Section 6: 9 Quality Control Gates

> Mỗi record phải vượt qua **tất cả 9 gates**. Bất kỳ gate nào reject thì record vào file rejected.
>
> Từ G1 kiểm tra choices duplicate, G2 parse JSON, G3-G4 kiểm tra answer/reasoning rỗng, G5 reasoning quá ngắn, G6 reasoning tiếng Anh (>30% từ Anh), G7 answer không khớp gold, G8 Syllogism trả lời "answerable" thay vì nội dung, đến G9 reasoning chung chung thiếu con số và thời hạn.
>
> Đây là lớp bảo vệ quan trọng — không có nó thì Student sẽ học từ data lỗi.

---

## Section 7: Implementation Roadmap

> Lộ trình 6 phase: chia data → pilot 100-300 samples → distill full train set → Student SFT → evaluate trên test gốc → production.
>
> Target: parse success ≥95%, answer match gold ≥98%, empty reasoning ≤2%, invalid answer ≤1%.

---

## Section 8: Why CPT?

> Chuyển sang phần Continual Pre-Training. Các SLM như Qwen, Gemma ở 1B-3B rất mạnh ngôn ngữ chung, nhưng khi hỏi về **pháp luật Việt Nam** — tên luật, số hiệu nghị định — chúng trả lời mơ hồ hoặc sai.
>
> CPT dạy thêm model về domain pháp lý mà **không làm mất kiến thức chung**. Dùng objective next-token prediction — model đọc văn bản luật và học cách dự đoán từ tiếp theo.
>
> Mục tiêu: bổ sung vocabulary, cấu trúc văn bản, và tri thức pháp luật Việt Nam — trước khi SFT.

---

## Section 9: Data Source — VLSP 2025

> Dataset **VLSP2025-LegalSML/legal-pretrain** trên HuggingFace. Tổng cộng 24 shards, dự án chỉ dùng **2 shards** cho thử nghiệm — 8,066 văn bản pháp luật, ước tính ~92M tokens.

---

## Section 10: Data Processing Pipeline

> Pipeline xử lý 6 bước: Load parquet → Deduplication (loại trùng theo Id, DocIdentity, content prefix) → Normalize (v1 regex strip hoặc v2 markitdown giữ cấu trúc) → Filter (token length 100-50K, blank ratio <0.6, non-VN ratio <0.3) → Format (gắn metadata header) → Save parquet.

---

## Section 11: Clean v1 vs v2

> So sánh 2 phiên bản cleaning:
>
> **v1 Regex Strip:** giữ 6,364 docs, ~35.6M tokens, **không giữ cấu trúc**.
> **v2 Markitdown:** giữ **7,958 docs (+25%)**, ~46.6M tokens (+31%), **giữ bảng và heading**.
>
> v2 tốt hơn rõ rệt — nhiều data hơn và giữ được cấu trúc văn bản luật vốn rất quan trọng (bảng phụ lục, heading điều khoản).

---

## Section 12: Train/Test Split

> Chia đơn giản: **7,560 train (95%)** và **398 test (5%)**. Đủ cho CPT vì objective chỉ là next-token prediction, không cần validation phức tạp.

---

## Section 13: LoRA & QLoRA for CPT

> So sánh 3 approach: Full Fine-Tuning tốn VRAM rất cao nhưng domain knowledge tốt nhất, có nguy cơ catastrophic forgetting cao. **LoRA** VRAM trung bình, forgetting thấp. **QLoRA** VRAM thấp nhất, forgetting thấp.
>
> Tham số khuyến nghị cho CPT-LoRA: rank 64-128, target cả attention + MLP + embed_tokens + lm_head, learning rate 1e-4 ~ 5e-5, 2-3 epochs cho dataset nhỏ ~46M tokens.

---

## Section 14: CPT Metrics — From Loss to Reality

> Cách đọc metrics: Loss 1.0 → PPL 2.7 (model gần như chắc chắn), Loss 2.0 → PPL 7.4 (còn không chắc), Loss 3.5 → PPL 33 (đoán mò).
>
> Ba tầng đánh giá: **CrossEntropyLoss** trong training — model đang học không? **Perplexity** sau CPT — hiểu domain tốt hơn chưa? **QA Accuracy** sau SFT — có ích thực tế không?

---

## Section 15: Experiment Setup

> So sánh 2 run: **LoRA (bf16)** vs **QLoRA (4-bit NF4)** trên Qwen3-4B, cùng rank 64, cùng learning rate 1e-4, cùng 1 epoch.
>
> Khác biệt chính: QLoRA dùng alpha=128 (scale=2.0) để bù cho quantization noise, trong khi LoRA dùng alpha=64 (scale=1.0). Batch size effective đều là 16 nhưng cấu hình khác: LoRA 4×4, QLoRA 8×2.

---

## Section 16: Training Dynamics

> Nhìn vào bảng Loss Trajectory: cả hai bắt đầu từ ~1.04, giảm nhanh xuống ~0.60 trong 1 epoch.
>
> QLoRA có grad_norm trung bình cao hơn (0.2256 vs 0.1857) — update có thể ồn ào hơn do quantization + alpha cao. Nhìn vào WandB charts thấy rõ: train/loss cả hai giảm song song, learning rate theo cosine schedule, grad_norm của LoRA (xanh lá) có spike rõ ở cuối.

---

## Section 17: CPT Results

> **LoRA:** Perplexity **1.86**, eval loss 0.6197, run hoàn thành.
> **QLoRA:** Perplexity **1.83** (tốt hơn nhẹ), eval loss 0.6034, nhưng **run bị failed**.
>
> QLoRA cho số đẹp hơn nhưng không đáng tin vì run crashed. Cần rerun với checkpoint/monitor. Recommendations: thêm eval_strategy=steps, thêm 1-2 epoch, sweep alpha 64/128 để tách ảnh hưởng quantization và alpha.

---

## Section 18: From CPT to SFT

> Sau CPT, model đã quen vocabulary pháp lý — nó biết "nghị định", "điều khoản" là gì. Nhưng nó **chưa biết cách trả lời câu hỏi**.
>
> SFT giải quyết bằng cách train trên cặp instruction-response. CPT cho model biết ngôn ngữ luật → SFT cho model biết cách dùng kiến thức đó để trả lời.

---

## Section 19: SFT Configuration

> Base model: **qwen3-1.7b-legal-pretrain** — model đã qua CPT. QLoRA rank 16 (nhỏ hơn CPT vì task đơn giản hơn), alpha 32, 4-bit NF4.
>
> Batch effective 32, learning rate 2e-4 với cosine warmup 10%, 3 epochs, optimizer adamw_8bit tiết kiệm ~50% VRAM.
>
> Cách đọc WandB: train/loss giảm = model đang học, eval/loss giảm = không overfit, eval/loss tăng = **OVERFIT**, grad_norm 0.1-1.0 = ổn định.

---

## Section 20: 3 Task Evaluation

> Đánh giá trên 3 task từ đơn giản đến phức tạp:
>
> **NLI:** 188 samples, binary — "Có liên quan" hay "Không liên quan". Baseline 55-60%, target >85%.
>
> **Multi-choice:** 133 samples, chọn A/B/C/D. Random baseline 25%, target >70%.
>
> **Syllogism:** 112 samples, open-ended QA dựa trên căn cứ pháp luật. Đo bằng ROUGE-L, target >0.40. Lưu ý ROUGE thấp không nhất thiết sai — model có thể paraphrase đúng nhưng khác wording.

---

## Section 21: Thinking Mode

> Qwen3 có 2 chế độ: **Thinking ON** inject prefix `<think>` — model viết reasoning chain trước rồi mới trả lời. Chậm hơn nhưng chính xác hơn cho câu hỏi phức tạp.
>
> **Thinking OFF** — model bỏ qua reasoning và trả lời thẳng. Nhanh, phù hợp câu hỏi đơn giản.
>
> Test set: 75% thinking=True, 25% thinking=False. Khi tính metrics, phần think bị strip — chỉ đánh giá câu trả lời cuối cùng.

---

## Section 22: SFT Training Results

> Train loss giảm ổn định: 1.5037 → 0.4703 (min) → 0.5220 (end). Eval loss dao động nhẹ quanh ~0.74 — **không có dấu hiệu overfit rõ ràng** trong 3 epoch.
>
> Dataset: 2,603 train / 322 valid / 325 test, phân bố đều qua 3 task types.

---

## Section 23: 3 Task Results

> **NLI:** Accuracy **0.6809**, F1 0.4819. Unknown 11% — tức 16/141 câu không extract được answer.
>
> **MC:** Accuracy chỉ **0.16** — nhưng **64% câu là unknown** (64/100 không extract được đáp án). Đây là vấn đề parser, không phải model không biết.
>
> **Syllogism:** ROUGE-L **0.3897** — gần target 0.40.
>
> **Breakdown by thinking:** thinking=false **tốt hơn** thinking=true trên cả NLI (+12% acc) và Syllogism (+0.03 RL). Model ổn định hơn khi trả lời thẳng — không cần chain-of-thought cho các task này.

---

## Section 24: Cross-Analysis & Recommendations

> **Vấn đề chính:** MC accuracy rất thấp nhưng nguyên nhân chính là parser — 64% không extract được. NLI accuracy trên tập có đáp án thực ra là ~0.768 — khá hơn nhiều so với 0.68 ban đầu.
>
> **Khuyến nghị:** Fix parser MC/NLI bằng constrained decoding hoặc regex extract. Dùng direct mode (thinking off) khi deploy. Bổ sung data hoặc re-weight cho MC. Thêm eval_strategy=steps để monitor chi tiết hơn.
>
> Thinking flag trong data là thuộc tính dataset, không phải inference mode — cần cẩn thận khi kết luận.

---

## Section 25: Full Pipeline Overview

> Tổng kết toàn bộ pipeline:
>
> **VLSP 2023** → 3,428 câu hỏi → Knowledge Distillation với Gold-Conditioned Reasoning + 9 Quality Gates
> → **VLSP 2025** → 7,958 văn bản pháp luật → Continual Pre-Training (Qwen3-4B, LoRA/QLoRA, PPL 1.83-1.86)
> → Supervised Fine-Tuning (Qwen3-1.7B, QLoRA r=16, 3 epochs)
> → NLI 68% | MC 16% (cần fix parser) | Syllogism 0.39 ROUGE-L
>
> Từ 3,428 câu hỏi thô và 7,958 văn bản pháp luật, pipeline đã xây dựng được model có khả năng hiểu ngôn ngữ pháp lý Việt Nam. NLI và Syllogism cho kết quả khả quan. MC cần fix parser trước khi đánh giá thật sự. Hành trình còn dài, nhưng nền tảng đã vững.
>
> Cảm ơn mọi người. Mình sẵn sàng nhận câu hỏi.
