# 02 — Human review chất lượng data distilled

> Đánh giá 3 sample thực tế từ góc nhìn data quality: NLI `01153`, NLI `00393`, Syllogism `00711`.

---

## Tổng quan

| Sample | Parser | Nội dung Teacher | Vấn đề chính | Train readiness |
|---|---|---|---|---|
| NLI `01153` | Pass | Đúng ý | Reasoning tiếng Anh, schema NLI/choices lỗi | Trung bình thấp |
| NLI `00393` | Pass | Đúng ý | Choices duplicate nghiêm trọng, label semantics sai | Trung bình thấp |
| Syllo `00711` | Pass | Quá ngắn | Reasoning nghèo hơn target gốc, objective sai | Thấp nếu dùng để SFT reasoning |

**Root cause:** Vấn đề không nằm ở Teacher — nằm ở **data schema và prompt objective**.

---

## Sample NLI `01153`

### Nội dung

- Câu hỏi: "Chuyển nhượng hối phiếu đòi nợ bằng ký chuyển nhượng là gì? Điều luật có liên quan không?"
- Điều luật: Điều 30 định nghĩa trực tiếp "chuyển nhượng bằng ký chuyển nhượng"
- Gold: `label=1` (relevant), Teacher: `"D"` (index 3) = đúng

### Điểm tốt

Teacher hiểu đúng nội dung. Answer D = index 3 = relevant = khớp gold.

### Vấn đề

**Reasoning tiếng Anh:**
```
"The legal document provided is directly related..."
```

Với legal SLM tiếng Việt → Student sẽ học style trả lời tiếng Anh/trộn ngôn ngữ.

**Schema NLI choices lỗi (phiên bản cũ):**
```json
"choices": ["không liên quan", "không liên quan và không thể...", "không liên quan", "có liên quan"]
```
A và C duplicate. `label_semantics` ghi `0=relevant` nhưng choices[0] là "không liên quan" → mâu thuẫn.

### Verdict

| Hạng mục | Đánh giá |
|---|---|
| Gold label match | Pass |
| Legal correctness | Pass |
| Parser validity | Pass |
| Reasoning language | Fail |
| NLI schema | Fail (phiên bản cũ) |
| Train readiness | Không dùng nguyên dạng |

---

## Sample NLI `00393`

### Nội dung

- Điều luật: "Dùng a-xít nguy hiểm hoặc hóa chất nguy hiểm"
- Câu hỏi: "Pháp nhân phi thương mại có bị truy cứu trách nhiệm hình sự không?"
- Gold: D — không thể dùng làm căn cứ
- Teacher reasoning tiếng Việt, ngắn, đúng nhãn ✓

### Vấn đề nghiêm trọng

Choices (phiên bản cũ):
```json
["có liên quan và có thể dùng làm căn cứ",
 "có liên quan và có thể dùng làm căn cứ",
 "có liên quan và có thể dùng làm căn cứ",
 "không thể dùng làm căn cứ"]
```

A/B/C giống hệt nhau → Student học shortcut: "nếu 3 options đầu duplicate thì chọn cái còn lại" thay vì học semantic relevance thật.

### Verdict

| Hạng mục | Đánh giá |
|---|---|
| Legal correctness | Pass |
| Language | Pass |
| Choices quality | **Fail nghiêm trọng** |
| Train readiness | Không dùng nguyên dạng |

---

## Sample Syllogism `00711`

### Nội dung

- Câu hỏi: "Thời hạn giải quyết thủ tục gộp sổ bảo hiểm xã hội là bao lâu?"
- Target.answer gốc: "Không quá 10 ngày; nếu cần xác minh thì không quá 45 ngày và phải thông báo"
- Teacher output:
```json
{
  "teacher_answer": "answerable",
  "teacher_reasoning": "Vì câu hỏi đã được trả lời bằng cách trích dẫn quy trình và thời hạn cụ thể."
}
```

### Vấn đề

Teacher bị reject bởi `parser.py`:
- `answer = "answerable"` → thuộc `ANSWERABILITY_LABELS` → gate 8 reject
- Reasoning không chứa "10 ngày" hay "45 ngày" → `_syllogism_keeps_target_facts()` = False → gate 9 reject

**Objective sai:** Prompt cũ biến syllogism thành classification "có trả lời được không?" thay vì "hãy trả lời pháp lý".

### Output tốt cần có

```json
{
  "answer": "Thời hạn gộp sổ BHXH là không quá 10 ngày kể từ ngày nhận đủ hồ sơ; nếu cần xác minh quá trình đóng ở tỉnh khác thì không quá 45 ngày và phải thông báo cho người lao động.",
  "reasoning": "Điều 29 quy định việc cấp lại sổ do gộp sổ BHXH không quá 10 ngày; trường hợp cần xác minh thì tối đa 45 ngày."
}
```

### Verdict

| Hạng mục | Đánh giá |
|---|---|
| Parser | Reject (gate 8 + 9) |
| Task objective | Fail — classification thay vì QA |
| Training usefulness | Thấp |

---

## Root causes tổng hợp

### Root cause 1: NLI schema sai từ converter (đã sửa)

- Choices duplicate trong `vilawqa_nli.parquet` raw
- `label_semantics` hard-code không khớp thực tế choices
- `parser.py` phát hiện và reject qua `_has_duplicate_choices()` và `_nli_semantics_consistent()`
- **Sửa:** Normalize NLI sang binary `relevant/irrelevant`, 2 choices sạch

### Root cause 2: `teacher_valid = true` ≠ sample chất lượng tốt

Parser chỉ check syntax/label gate. Chưa check:
- Reasoning có tiếng Việt không
- Choices có duplicate không
- Label semantics có khớp choices không
- Reasoning có đủ key facts không

### Root cause 3: Syllogism objective sai trong prompt cũ

Cần đổi prompt sang:
```
Đáp án/câu trả lời đúng đã cho là:
{target.answer}

Hãy viết lại lời giải pháp lý ngắn gọn, chính xác, giữ đủ con số/thời hạn quan trọng.
```

---

## Quality gates cần thêm vào filter

```
NLI:
- reject nếu choices duplicate ratio cao
- reject nếu label_semantics mâu thuẫn với choices
- reject nếu reasoning tiếng Anh > 30%
- reject nếu reasoning < 30 ký tự

Syllogism:
- reject nếu teacher_answer chỉ là answerable/not answerable
- reject nếu reasoning không chứa key numbers từ target.answer
- reject nếu key term overlap < 15%
- reject nếu reasoning quá chung chung (generic phrases)

Multi-choice:
- reject nếu teacher_answer không phải single letter A/B/C/D
- reject nếu teacher_answer != gold_label
- reject nếu reasoning tiếng Anh
- reject nếu reasoning < 20 ký tự
```

---

## Go/No-go trước khi train Student

```
parse_success_rate  >= 95%
answer_match_gold   >= 98%
empty_reasoning_rate <= 2%
invalid_answer_rate  <= 1%
manual review 50 records: reasoning ổn, tiếng Việt, đủ facts
```
