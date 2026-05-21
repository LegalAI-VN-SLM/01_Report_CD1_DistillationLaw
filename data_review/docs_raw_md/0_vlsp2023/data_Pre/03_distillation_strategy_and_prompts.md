# 03 — Chiến lược distillation và prompt mẫu

> Quyết định làm lại distillation theo hướng gold-conditioned reasoning. Prompt mẫu cho từng task.

---

## Quyết định: Làm lại distillation trước khi train Student

Lý do: Data SeaLLMs v1 (20 mẫu) chỉ là output debug.

```
parse_success_rate   = 80%
teacher accuracy     ≈ 40%
invalid/missing ans  = 20%
single-letter clean  = 25%
```

Với chất lượng này, Student sẽ học nhiễu: answer sai, format lộn xộn, reasoning yếu/mâu thuẫn.

---

## Thay đổi chiến lược: Gold-conditioned reasoning

### Trước (tự chọn đáp án):
```
Question + choices → SeaLLMs chọn answer + reasoning
```

### Sau (gold-conditioned):
```
Question + choices + đáp án đúng → SeaLLMs sinh reasoning
```

SeaLLMs-7B không đủ tin cậy để làm answer oracle (accuracy ~40%). Nhưng **vẫn rất hữu ích để viết giải thích tiếng Việt nếu đã biết đáp án đúng**.

---

## Pipeline mới

```
raw_unified.jsonl
   ↓
split_train_val_test.py (stratify theo task_type)
   ↓
split_train_v1.jsonl (80%)
split_val_v1.jsonl   (10%)
split_test_v1.jsonl  (10%)
   ↓
chunked_distill.py chỉ chạy trên split_train_v1.jsonl
   ↓
build_prompt_with_gold_answer() theo task_type
   ↓
SeaLLMs sinh reasoning JSON (không tự chọn đáp án)
   ↓
parser.py quality gates
   ↓
distilled_train_clean.jsonl
distilled_train_rejected.jsonl
   ↓
train Qwen3-4B Student
   ↓
evaluate trên split_test_v1.jsonl (không distill, không cho Teacher nhìn)
```

---

## Prompt mẫu theo task

### Multi-choice

```
<|im_start|>system
Bạn là chuyên gia pháp luật Việt Nam.
Chỉ trả về JSON hợp lệ theo schema được yêu cầu. Không thêm text ngoài JSON.
Bắt buộc dùng tiếng Việt cho tất cả giải thích.
<|im_end|>

<|im_start|>user
Câu hỏi:
{question}

Các lựa chọn:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Đáp án đúng là: {gold_letter}. {gold_answer_text}

Yêu cầu:
- Không đổi đáp án.
- Giải thích bằng tiếng Việt vì sao {gold_letter} đúng.
- Giải thích ngắn vì sao các phương án còn lại không phù hợp.

Trả về JSON:
{
  "answer": "{gold_letter}",
  "reasoning": "...",
  "elimination": {
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  }
}
<|im_end|>

<|im_start|>assistant
```

**Parser check:** `parsed["answer"] == gold_letter`

---

### NLI (binary relevance)

```
<|im_start|>system
Bạn là chuyên gia pháp luật Việt Nam.
Chỉ trả về JSON hợp lệ. Bắt buộc dùng tiếng Việt.
<|im_end|>

<|im_start|>user
Điều luật:
{legal_document}

Câu hỏi:
{question}

Nhãn đúng là: {gold_label}.  (relevant = điều luật liên quan / irrelevant = không liên quan)

Hãy giải thích bằng tiếng Việt vì sao điều luật này {relevant/không} liên quan đến câu hỏi.

Trả về JSON:
{
  "label": "{gold_label}",
  "reasoning": "..."
}
<|im_end|>

<|im_start|>assistant
```

**Parser check:** `parsed["label"] in {"relevant", "irrelevant"}` và khớp gold.

---

### Syllogism (QA + rationale)

```
<|im_start|>system
Bạn là chuyên gia pháp luật Việt Nam.
Chỉ trả về JSON hợp lệ. Bắt buộc dùng tiếng Việt. Giữ nguyên các con số và thời hạn pháp lý.
<|im_end|>

<|im_start|>user
Câu hỏi pháp lý:
{question}

Điều luật tham chiếu:
{reference_texts}

Câu trả lời chuẩn đã có:
{target.answer}

Hãy viết lại câu trả lời pháp lý ngắn gọn, chính xác, giữ đủ các con số và thời hạn quan trọng.
Sau đó giải thích căn cứ pháp lý.

Trả về JSON:
{
  "answer": "...",
  "reasoning": "Căn cứ [Điều X Nghị định Y]..."
}
<|im_end|>

<|im_start|>assistant
```

**Parser check:** `answer` không phải answerability label, `len(answer) >= 30`, chứa key numbers từ target.

---

## Quality filter — ngưỡng go/no-go

```
parse_success_rate  >= 95%
answer_match_gold   >= 98%
empty_reasoning     <= 2%
english_reasoning   <= 2%
invalid_answer      <= 1%
```

Nếu chưa đạt: sửa prompt/parser, chưa train Student.

---

## Khi nào được train Student

Sau khi:
1. Chạy 100 samples thử — check parse_success, reasoning_length, language
2. Report sạch theo ngưỡng trên
3. Manual review 50 records: reasoning đúng pháp lý, tiếng Việt, đủ facts
4. Có `distilled_train_clean.jsonl` với ít nhất vài nghìn records sạch

Training config:
```python
MODEL_NAME = "unsloth/Qwen3-4B-bnb-4bit"
MAX_SEQ_LENGTH = 4096
LORA_R = 32; LORA_ALPHA = 64; LORA_DROPOUT = 0.0
LR = 1.5e-4; EPOCHS = 2-3; BATCH_SIZE = 4; GRAD_ACCUM = 4
BF16 = True
```

---

## Lệnh chạy distillation

```bash
# Test 5 samples
python -m slm_data_distillation_law.teacher.pipeline \
  --input data/processed/jsonl/split_train_v1.jsonl \
  --output-dir data/processed/jsonl \
  --limit 5 \
  --overwrite

# Test 100 samples
python -m slm_data_distillation_law.teacher.pipeline \
  --input data/processed/jsonl/split_train_v1.jsonl \
  --output-dir data/processed/jsonl \
  --limit 100 \
  --overwrite

# Full run (chỉ khi 2 bước trên đạt ngưỡng)
python -m slm_data_distillation_law.teacher.pipeline \
  --input data/processed/jsonl/split_train_v1.jsonl \
  --output-dir data/processed/jsonl \
  --overwrite
```

---

## Sau khi distill: phân tích chất lượng

```bash
# Xem thống kê records/task, field rỗng, field thiếu
python data_distillation_final/analyze_distilled_chunks.py --split train

# Gom data sạch thành file output
python data_distillation_final/merge_non_empty_by_task.py --split train
```

Output:
```
data/v1/train/
├── vilawqa_nli_train.jsonl
├── vilawqa_mc_train.jsonl
└── vilawqa_syllo_train.jsonl
```
