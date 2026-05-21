# 03 — Tại sao `teacher_reasoning` / `teacher_answer` bị rỗng?

> Giải thích chi tiết dựa trên source code `parser.py`, `generator.py`, `pipeline.py`.

---

## Tổng quan luồng sinh data

```
generator.py
   generate_batch(prompts)
   → raw_text (string JSON từ model)
        ↓
parser.py
   parse_raw(raw_text, task_type, gold_label, sample)
   → ParsedOutput(is_valid, answer, reasoning, error_msg)
        ↓
pipeline.py
   build_distilled_record(...)
   → distilled JSONL record
```

Khi `is_valid=False`, record vẫn được lưu vào `distilled_train_rejected.jsonl` với `teacher_answer=""`, `teacher_reasoning=""`, và `teacher_error` ghi rõ lý do.

---

## Các lý do cụ thể khiến `reasoning` hoặc `answer` rỗng

### Gate 1 — Validate sample trước khi parse

```python
# parser.py — _validate_sample_quality()
def _validate_sample_quality(self, task_type, sample):
    if task_type == "nli":
        if self._has_duplicate_choices(sample):
            return "nli choices contain duplicates"
        if not self._nli_semantics_consistent(sample):
            return "nli label_semantics inconsistent with binary relevance schema"
    return None
```

**Kết quả:** Nếu NLI có choices duplicate hoặc `label_semantics` không khớp chuẩn `0=irrelevant,1=relevant` → record bị reject ngay, `answer=""`, `reasoning=""`.

Đây là nguyên nhân chính khiến NLI bị reject hàng loạt — vì converter `to_jsonl.py` đang hard-code `label_semantics` sai và choices bị duplicate.

---

### Gate 2 — Parse JSON thất bại

```python
# parser.py — extract_json()
parsed_obj, error = self.extract_json(raw_text)
if parsed_obj is None:
    return ParsedOutput(is_valid=False, answer="", reasoning="", error_msg=error)
```

**Nguyên nhân thường gặp:**
- Model sinh văn xuôi tự do thay vì JSON (`The exact answer text is: D...`)
- Model sinh JSON không hợp lệ (thiếu dấu `}`, key sai)
- Model sinh thêm text trước/sau JSON
- Prompt chưa đủ chặt để ép JSON output

`extract_json()` thử 2 cách: parse toàn bộ text, sau đó regex tìm JSON block. Nếu cả 2 đều fail → `answer=""`, `reasoning=""`.

---

### Gate 3 — `answer` rỗng sau parse

```python
if not answer:
    return ParsedOutput(is_valid=False, answer="", reasoning="", error_msg="answer is empty")
```

**Nguyên nhân:** Model sinh JSON nhưng field `answer` (hoặc `label` với NLI) không có giá trị.

---

### Gate 4 — `reasoning` rỗng sau parse

```python
if not reasoning:
    return ParsedOutput(is_valid=False, answer=answer, reasoning="", error_msg="reasoning is empty")
```

**Nguyên nhân:** Model sinh `{"answer": "B", "reasoning": ""}` hoặc bỏ key `reasoning` hoàn toàn.

Với reasoning rỗng: `answer` được giữ lại trong record nhưng `teacher_valid=false`, record vào rejected.

---

### Gate 5 — Reasoning quá ngắn

```python
task_min_reasoning_len = self.min_reasoning_len  # default 20
if task_type == "nli":
    task_min_reasoning_len = max(task_min_reasoning_len, 30)
elif task_type == "syllogism":
    task_min_reasoning_len = max(task_min_reasoning_len, 40)

if len(reasoning) < task_min_reasoning_len:
    return ParsedOutput(is_valid=False, ...)
```

**Ngưỡng tối thiểu:**
- `multi_choice`: 20 ký tự
- `nli`: 30 ký tự
- `syllogism`: 40 ký tự

Reasoning quá ngắn (ví dụ chỉ "Đúng." hay "B là đáp án.") → reject.

---

### Gate 6 — Reasoning tiếng Anh

```python
if self._english_word_ratio(reasoning) > 0.3 and not self._looks_vietnamese(reasoning):
    return ParsedOutput(
        is_valid=False,
        error_msg="reasoning appears to be English",
    )
```

Nếu > 30% từ trong reasoning là tiếng Anh (theo danh sách `ENGLISH_STOPWORDS`) **và** không tìm thấy hint tiếng Việt → reject.

**Ví dụ thực tế từ data:**
```
"teacher_reasoning": "The legal document provided is directly related..."
```
→ bị reject vì tiếng Anh.

**Lưu ý:** SeaLLMs-v3 được train đa ngôn ngữ, khi prompt tiếng Việt không đủ chặt, model có thể trả lời tiếng Anh.

---

### Gate 7 — Answer không khớp gold label

**Multi-choice:**
```python
if len(normalized_answer) == 1 and normalized_answer in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    answer_idx = ord(normalized_answer) - ord('A')
    if answer_idx != gold_label:
        return ParsedOutput(is_valid=False, error_msg=f"answer {answer} (idx {answer_idx}) != gold {gold_label}")
else:
    return ParsedOutput(is_valid=False, error_msg=f"answer not a valid letter: {answer}")
```

Reject nếu:
- Answer là `E`, `F`... (ngoài A/B/C/D)
- Answer không phải single letter
- Answer đúng chữ nhưng sai gold label

**NLI:**
```python
expected = "relevant" if gold_label == 1 else "irrelevant" if gold_label == 0 else ""
if normalized_answer not in {"relevant", "irrelevant"}:
    return ParsedOutput(is_valid=False, error_msg=f"nli label not binary relevance: {answer}")
if expected and normalized_answer != expected:
    return ParsedOutput(is_valid=False, error_msg=f"label {answer} != gold {expected}")
```

NLI chỉ chấp nhận `relevant` hoặc `irrelevant`. Nếu model trả `D`, `có liên quan`, `yes`... → reject.

---

### Gate 8 — Syllogism: answer quá đơn giản hoặc quá ngắn

```python
# Reject nếu answer chỉ là answerability label
ANSWERABILITY_LABELS = {"answerable", "not answerable", "unanswerable"}
if answer.lower() in ANSWERABILITY_LABELS:
    return ParsedOutput(is_valid=False, error_msg="syllogism answer is answerability label, not legal QA answer")

# Reject nếu answer quá ngắn
if len(answer) < 30:
    return ParsedOutput(is_valid=False, error_msg=f"syllogism answer too short: {len(answer)} < 30")
```

**Nguyên nhân:** Prompt cũ hỏi teacher "câu này có trả lời được không?" → teacher trả `"answerable"` → bị reject.

Syllogism cần answer là **nội dung pháp lý thực sự**, không phải classification.

---

### Gate 9 — Syllogism: reasoning bỏ qua facts quan trọng

```python
GENERIC_SYLLO_PHRASES = (
    "câu hỏi đã được trả lời",
    "tài liệu tham khảo",
    "kết luận hợp lệ",
    "có thể trả lời",
    "đã nêu rõ",
)

if any(phrase in reasoning.lower() for phrase in GENERIC_SYLLO_PHRASES) \
        and not self._syllogism_keeps_target_facts(answer, reasoning, sample):
    return ParsedOutput(is_valid=False, error_msg="syllogism reasoning is too generic...")
```

`_syllogism_keeps_target_facts()` kiểm tra:
- Output phải chứa **các con số** từ `target.answer` (ví dụ "10 ngày", "45 ngày")
- Overlap key terms >= 15%

**Ví dụ thực tế bị reject:**
```
target.answer: "Không quá 10 ngày; trường hợp xác minh: không quá 45 ngày"
teacher_reasoning: "Câu hỏi đã được trả lời bằng cách trích dẫn quy trình trong tài liệu tham khảo."
```
→ Reasoning generic, thiếu "10 ngày" và "45 ngày" → reject.

---

## Bảng tóm tắt tất cả lý do reject

| Lý do | `teacher_error` trong record | Nguyên nhân gốc |
|---|---|---|
| NLI choices duplicate | `"nli choices contain duplicates"` | converter `to_jsonl.py` — choices lặp |
| NLI semantics sai | `"nli label_semantics inconsistent..."` | converter hard-code sai semantics |
| Không parse được JSON | `"Could not parse JSON from output"` | Model sinh văn xuôi, không JSON |
| `answer` rỗng | `"answer is empty"` | Model không điền field answer |
| `reasoning` rỗng | `"reasoning is empty"` | Model không điền field reasoning |
| Reasoning quá ngắn | `"reasoning too short: X < Y"` | Model sinh reasoning 1–2 từ |
| Reasoning tiếng Anh | `"reasoning appears to be English"` | Prompt không ép tiếng Việt |
| Answer sai gold | `"answer B (idx 1) != gold 2"` | Teacher chọn sai đáp án |
| Answer không hợp lệ | `"answer not a valid letter: E"` | Teacher sinh đáp án ngoài schema |
| NLI label không binary | `"nli label not binary relevance: D"` | Prompt cũ chưa ép binary relevance |
| Syllo answer là answerability | `"syllogism answer is answerability label..."` | Prompt cũ hỏi "có trả lời được không?" |
| Syllo answer quá ngắn | `"syllogism answer too short: 15 < 30"` | Answer chỉ 1 câu ngắn |
| Syllo reasoning generic | `"syllogism reasoning is too generic..."` | Thiếu con số/facts từ target.answer |
| Syllo reasoning thiếu facts | `"syllogism output misses key facts..."` | Overlap key terms < 15% |

---

## Nguyên nhân gốc (Root causes)

### Root cause 1: NLI schema sai từ converter

File `to_jsonl.py` hard-code `label_semantics = "0=relevant, 1=irrelevant, 2=relevant, 3=relevant"` và để nguyên choices từ raw parquet — nhiều bộ choices bị duplicate.

`parser.py` detect được và reject toàn bộ NLI có vấn đề này. **Sửa ở converter, không phải ở parser.**

### Root cause 2: Prompt chưa ép JSON + tiếng Việt

`generator.py` dùng system prompt:
```python
SYSTEM_PROMPT = "Bạn là một chuyên gia pháp luật Việt Nam. Hãy phân tích..."
```

System prompt không ép JSON format → model tự do sinh văn xuôi → parse fail.

**Cần thêm vào prompt:**
```
Chỉ trả về JSON hợp lệ theo schema sau. Không thêm text ngoài JSON.
Bắt buộc dùng tiếng Việt cho reasoning.
```

### Root cause 3: Syllogism objective sai

Prompt cũ hỏi model "câu này có trả lời được không?" → model trả `answerable` → bị gate 8 reject.

Cần đổi sang: "đáp án đúng là X, hãy viết lại lời giải pháp lý".

---

## Cách đọc `teacher_error` trong rejected file

```python
# Mỗi record rejected có dạng:
{
  "teacher_answer": "B",      # có thể có hoặc rỗng
  "teacher_reasoning": "",     # thường rỗng nếu fail sớm
  "teacher_valid": false,
  "teacher_error": "reasoning too short: 15 < 30",  # ← đọc cái này
  ...
}
```

Để debug, chạy:
```bash
python analyze_distilled_chunks.py --split train
```

Section 2 và 3 sẽ cho thấy field nào bị thiếu/rỗng nhiều nhất và theo task nào.
