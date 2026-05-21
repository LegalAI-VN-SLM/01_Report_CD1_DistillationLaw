# 01 — Đánh giá kiến trúc pipeline distillation

> Đánh giá toàn bộ pipeline từ góc nhìn senior ML. File này tổng hợp kiến trúc hiện tại, điểm mạnh/yếu và lộ trình đề xuất.

---

## Executive Summary

Pipeline đang dùng Teacher model lớn để sinh `answer + reasoning`, sau đó fine-tune Student model nhỏ cho bài toán legal reasoning tiếng Việt. Hướng đi đúng.

**Chốt kiến trúc:**

```
Teacher chính : CMC-AI-Legal-32B
Teacher phụ  : SeaLLMs-v3-7B-Chat
Student chính : Qwen3-4B-Base hoặc Qwen3-4B legal-pretrain
Training      : QLoRA / LoRA bằng Unsloth
Evaluation    : split train/test từ raw_unified.jsonl TRƯỚC khi distill
```

**Điểm quan trọng nhất:** Không được chạy toàn bộ `raw_unified.jsonl` qua Teacher rồi mới chia train/test. Phải chia trước, chỉ distill trên train. Nếu không, Student sẽ bị data leakage và accuracy sẽ ảo.

---

## Kiến trúc pipeline hiện tại

```
raw_unified.jsonl
   ↓
prompting.py         ← build_user_prompt() theo task_type
   ↓
Teacher inference    ← generator.py (ChatML format, batch generation)
   ↓
parser.py            ← extract JSON, quality gates
   ↓
distilled_train_clean.jsonl
distilled_train_rejected.jsonl
   ↓
Student SFT / QLoRA
```

**4 module chính:**

| Module | File | Nhiệm vụ |
|---|---|---|
| Phase 1 | `loader.py` | Load model + tokenizer |
| Phase 2 | `generator.py` | Batch inference với ChatML format |
| Phase 3 | `parser.py` | Extract JSON, áp quality gates |
| Phase 4 | `pipeline.py` | Orchestrate toàn bộ |

**Điểm mạnh:**
- Modular, tách rõ loader / generator / parser / pipeline
- Có `--model-id`, dễ thay backend
- Output schema đầy đủ: `teacher_answer`, `teacher_reasoning`, `teacher_valid`, `teacher_error`
- Có tách clean / rejected để debug

**Điểm cần bổ sung:**
- Resume checkpoint cho distillation dừng giữa chừng
- Structured logging (Weights & Biases / MLflow)
- Dataset versioning
- Eval tự động sau mỗi run

---

## Lựa chọn model

### Teacher

| Model | Vai trò | Ưu tiên |
|---|---|---|
| CMC-AI-Legal-32B | Teacher chính / judge | Rất cao |
| SeaLLMs-v3-7B-Chat | Teacher phụ / baseline tiếng Việt | Cao |
| Qwen2.5-7B-Instruct | Teacher phụ reasoning | Trung bình–cao |

SeaLLMs-7B nên được dùng để **sinh reasoning khi đã biết gold answer**, không phải để tự chọn đáp án (accuracy sample thấp ~40–50%).

### Student

```
Qwen3-4B-Base / legal-pretrain   ← Student chính
Qwen3-4B-Instruct                ← Chỉ dùng để debug/test nhanh, không train chính
```

---

## Chiến lược distillation

Chiến lược phù hợp: **Level 2 — answer + reasoning SFT**.

```json
{
  "teacher_answer": "B",
  "teacher_reasoning": "Theo Điều ...",
  "teacher_confidence": 0.95,
  "meta": { "teacher_model": "CMC-OPENAI/CMC-AI-Legal-32B" }
}
```

> `teacher_confidence = 0.95` hiện là placeholder, không phải confidence từ logits. Không dùng để weight loss.

Ưu tiên đúng:
1. Parse đúng answer
2. Reasoning ngắn, sạch, đúng format tiếng Việt
3. Teacher answer khớp gold label
4. Loại sample teacher trả sai hoặc parse lỗi

---

## Quy trình evaluation đúng

```
raw_unified.jsonl
   ↓ split TRƯỚC
train.jsonl (80%)    val.jsonl (10%)    test.jsonl (10%)
   ↓                                        ↓
Teacher distill                     giữ nguyên, KHÔNG distill
   ↓
distilled_train_clean.jsonl
   ↓
Student SFT
   ↓
Evaluate Student trên test.jsonl
```

**4 lớp evaluation:**

| Lớp | Mục tiêu | Metric |
|---|---|---|
| Teacher accuracy | Teacher trả đúng gold label không | Accuracy |
| Parser quality | Có extract đúng A/B/C/D không | Parse success rate |
| Student accuracy | Student trả đúng trên held-out test | Accuracy / Macro-F1 |
| Reasoning quality | Giải thích có đúng luật, không hallucinate | Human eval / judge eval |

---

## Training config

### T4 16GB (debug only)

```python
MODEL_NAME = "unsloth/Qwen3-4B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LORA_R = 16; LORA_ALPHA = 16
LR = 2e-4; EPOCHS = 1; BATCH_SIZE = 1; GRAD_ACCUM = 16
FP16 = True; BF16 = False
```

### RTX 6000 Ada 48GB (training chính)

```python
MODEL_NAME = "unsloth/Qwen3-4B-bnb-4bit"
MAX_SEQ_LENGTH = 4096
LORA_R = 32; LORA_ALPHA = 64; LORA_DROPOUT = 0.0
LR = 1.5e-4; EPOCHS = 2-3; BATCH_SIZE = 4; GRAD_ACCUM = 4
BF16 = True
```

---

## Rủi ro kỹ thuật

| Rủi ro | Mức độ | Cách xử lý |
|---|---|---|
| Data leakage | Nghiêm trọng | Split trước, test không bao giờ qua teacher |
| Teacher reasoning sai nhưng nghe hợp lý | Cao | Filter: answer == gold AND parse OK AND reasoning đủ dài |
| Parser yếu | Trung bình | Lưu `teacher_raw_output`, log parse fail, unit test parser |
| Overfitting style teacher | Trung bình | Giới hạn format reasoning, dùng nhiều teacher |
| Không có baseline | Cao | Benchmark Qwen3-4B zero-shot trước khi train |

---

## Lộ trình 5 phase

```
Phase 0: Split data 80/10/10, stratify theo task_type
Phase 1: Teacher pilot 100–300 samples, check parse_success_rate >= 95%
Phase 2: Distill full train set, filter clean/rejected
Phase 3: Student SFT trên distilled_train_clean.jsonl
Phase 4: Evaluate trên test.jsonl gốc
Phase 5: Production hardening (resume, logging, versioning, model card)
```

**Go/No-go trước khi train Student:**

```
teacher_accuracy >= baseline student zero-shot
parse_success_rate >= 95%
reasoning không quá dài
```
