# 01 — Pipeline chuẩn bị raw data

> Giải thích các file trong bước convert raw → unified JSONL, trước khi đưa qua Teacher.

---

## Tổng quan

Các file ở bước này **chỉ convert raw data**, chưa sinh reasoning. Output là `raw_unified.jsonl` — đầu vào cho pipeline distillation.

```
data/raw/vilawqa.parquet
data/raw/vilawqa_nli.parquet
data/raw/vilawqa_syllo.parquet
         ↓
    to_jsonl.py
         ↓
raw_vilawqa_mc.jsonl
raw_vilawqa_nli.jsonl
raw_vilawqa_syllo.jsonl
         ↓
    merge 3 files
         ↓
raw_unified.jsonl (3,428 records)
```

---

## Phân bố dữ liệu

| Dataset | Task | Số mẫu |
|---|---|---|
| `vilawqa` | multi_choice | 1,007 |
| `vilawqa-nli` | nli | 1,421 |
| `vilawqa-syllo` | syllogism | 1,000 |
| **Tổng** | | **3,428** |

---

## Schema từng task

### Multi-choice (`vilawqa_mc`)

```json
{
  "id": "vilawqa_mc_00001",
  "dataset": "vilawqa",
  "task_type": "multi_choice",
  "input": {
    "question": "...",
    "choices": ["A...", "B...", "C...", "D..."]
  },
  "target": {
    "answer_index": 3,
    "answer_text": "...",
    "answer_format": "index"
  },
  "label": 3,
  "meta": { "split": "train" }
}
```

`label` và `target.answer_index` đều trỏ vào `choices[ground_truth]`.

**Vấn đề cần kiểm tra:** Có 2 record có `label = 7` và `label = 8` — bất thường với MCQ A/B/C/D (hợp lệ chỉ là 0–3). Cần inspect và sửa trước khi distill.

---

### NLI (`vilawqa_nli`)

```json
{
  "id": "vilawqa_nli_00001",
  "dataset": "vilawqa-nli",
  "task_type": "nli",
  "input": {
    "prompt": "### LEGAL DOCUMENT:\n...\n### QUESTION:\n...",
    "choices": ["Không liên quan", "Có liên quan"]
  },
  "target": {
    "answer_index": 1,
    "answer_text": "Có liên quan",
    "label": "relevant"
  },
  "label": 1,
  "meta": {
    "split": "train",
    "label_semantics": "0=irrelevant, 1=relevant",
    "schema_version": "nli_binary_v2"
  }
}
```

> **Ghi chú:** Schema NLI đã được chuẩn hóa sang binary `relevant/irrelevant` (label 0/1) thay vì MCQ 4 lựa chọn như phiên bản cũ. `parser.py` chỉ chấp nhận `label_semantics = "0=irrelevant,1=relevant"` — nếu converter sinh sai sẽ bị reject toàn bộ.

---

### Syllogism (`vilawqa_syllo`)

```json
{
  "id": "vilawqa_syllo_00001",
  "dataset": "vilawqa-syllo",
  "task_type": "syllogism",
  "input": {
    "question": "...",
    "refs": ["khoản 2 Điều 75 Nghị định 15/2021/NĐ-CP"],
    "reference_texts": [
      {
        "citation": "...",
        "content": "...",
        "text": "...",
        "meta": { "issuing_agency": "...", "sign_number": "..." }
      }
    ]
  },
  "target": {
    "answer": "Không quá 10 ngày kể từ ngày nhận đủ hồ sơ...",
    "rationale": "Căn cứ Điều 29..."
  },
  "label": 0,
  "meta": {
    "split": "train",
    "label_meaning": "0=answer_provided",
    "objective": "legal_qa_with_rationale",
    "teacher_ready": true
  }
}
```

`label = 0` với toàn bộ syllogism vì mọi sample đều có answer. `target.answer` và `target.rationale` là câu trả lời chuẩn đầy đủ — **Teacher phải sinh reasoning tương đương hoặc tốt hơn**.

---

## Vấn đề phiên bản NLI cũ (đã sửa)

Phiên bản converter cũ sinh NLI dạng MCQ 4 lựa chọn — có 2 vấn đề nghiêm trọng:

**Vấn đề 1:** Choices duplicate (A và C giống nhau, B và D giống nhau):
```json
"choices": [
  "không liên quan",
  "không liên quan và không thể dùng làm căn cứ",
  "không liên quan",          ← duplicate với A
  "có liên quan"
]
```

**Vấn đề 2:** `label_semantics` hard-code sai: `"0=relevant, 1=irrelevant, 2=relevant, 3=relevant"` không khớp với text choices.

→ `parser.py` detect và reject toàn bộ NLI dạng này qua gate `_has_duplicate_choices()` và `_nli_semantics_consistent()`.

---

## Quy trình đúng sau khi có raw_unified.jsonl

```
raw_unified.jsonl (3,428 records)
   ↓
split_train_val_test.py
stratify theo task_type
   ↓
split_train_v1.jsonl (80% ≈ 2,742 records)
split_val_v1.jsonl   (10% ≈  343 records)
split_test_v1.jsonl  (10% ≈  343 records)
   ↓
chunked_distill.py chỉ chạy trên split_train_v1.jsonl
   ↓
Teacher sinh reasoning
   ↓
distilled_train_clean.jsonl
distilled_train_rejected.jsonl
```

**Tuyệt đối không distill trên val/test.** Test set dùng duy nhất để evaluate Student.

---

## Validate split

Sau khi split, chạy:

```bash
python -m slm_data_distillation_law.teacher.validate_split
```

Kiểm tra:
- Số records mỗi split
- Phân bố task_type
- Không có JSON parse error
