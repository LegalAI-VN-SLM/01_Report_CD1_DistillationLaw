# 02 — Đánh giá chất lượng output SeaLLMs-v3-7B

> Review file `data_teacher_seallms_v1.jsonl` (20 mẫu multi_choice) từ góc nhìn senior ML/data quality.

---

## Kết quả kiểm tra nhanh

| Metric | Kết quả |
|---|---|
| Tổng mẫu | 20 |
| Parse được answer | 16 / 20 (80%) |
| Parse fail | 4 / 20 |
| Đúng so với gold label (trên parse được) | 8 / 16 (50%) |
| Accuracy toàn bộ 20 mẫu | 40% |
| `teacher_answer` là single letter sạch A/B/C/D | 5 / 20 (25%) |

**Kết luận:** File này phù hợp để **debug pipeline**, chưa đủ để train Student.

---

## Vấn đề phát hiện

### 1. Format answer không ổn định

Teacher sinh nhiều dạng output khác nhau:
```
C
D
The exact answer text is: D. ...
The exact answer is: A. ...
Cha mẹ nuôi theo quy định...
E                              ← ngoài A/B/C/D
```

Một số mẫu trả `E` dù bài chỉ có A/B/C/D.

### 2. `teacher_reasoning` không phải reasoning thật

Nhiều dòng chỉ copy lại answer:
```
The exact answer text is: D. ...
```

Distillation tốt cần dạng:
```
Đáp án: D
Giải thích:
1. Căn cứ quy định liên quan...
2. Đối chiếu với dữ kiện...
3. Loại các phương án sai...
4. Kết luận chọn D.
```

### 3. Teacher trả sai nhiều

| ID | Gold | Teacher | Nhận xét |
|---|---|---|---|
| `vilawqa_mc_00002` | D | C | Reasoning còn ghi cả C và D |
| `vilawqa_mc_00004` | D | E | Sinh đáp án ngoài option |
| `vilawqa_mc_00006` | D | A | Sai rõ |
| `vilawqa_mc_00011` | B | A | `reasoning` nói B nhưng `teacher_answer` là A — inconsistency |

### 4. `teacher_confidence = 0.95` cố định

Không phản ánh độ chắc chắn thật. Không dùng để weight loss.

### 5. Chỉ cover `multi_choice`

File chưa có NLI hoặc syllogism, chưa kết luận được gì về 2 task này.

---

## Chấm điểm tổng thể

| Hạng mục | Điểm | Nhận xét |
|---|---|---|
| Schema | 8/10 | Có đủ field cần thiết |
| Format consistency | 4/10 | Answer format chưa ổn |
| Teacher accuracy | 4/10 | ~40–50% trên sample này |
| Reasoning quality | 3/10 | Nhiều dòng không có reasoning thật |
| Train readiness | 3/10 | Chưa nên train Student |
| Debug usefulness | 8/10 | Tốt để tìm lỗi pipeline |

---

## Vai trò hợp lý của SeaLLMs

| Vai trò | Nên dùng? | Ghi chú |
|---|---|---|
| Generate reasoning khi đã biết gold answer | **Nên** | Rất phù hợp |
| Baseline tiếng Việt | **Nên** | Để so với Qwen3 |
| Teacher tự chọn đáp án | Không ưu tiên | Accuracy sample thấp |
| Judge cuối | Không nên | Dùng CMC-Legal-32B tốt hơn |

**Kết luận:** Đổi bài toán của SeaLLMs:

```
Từ: "hãy chọn đáp án"
Sang: "đáp án đúng là X, hãy giải thích vì sao"
```

---

## Prompt nên dùng

### Multi-choice (gold-conditioned)

```
<|im_start|>system
Bạn là chuyên gia pháp luật Việt Nam.
Bạn phải trả lời đúng JSON schema. Không thêm text bên ngoài JSON.
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
- Giải thích vì sao đáp án đúng là {gold_letter}.
- Giải thích ngắn vì sao các phương án còn lại không phù hợp.
- Trả về JSON:

{
  "answer": "{gold_letter}",
  "reasoning": "...",
  "elimination": {"A": "...", "B": "...", "C": "...", "D": "..."}
}
<|im_end|>

<|im_start|>assistant
```

### NLI (binary relevance)

```json
{
  "label": "relevant",
  "reasoning": "..."
}
```

### Syllogism (QA + rationale)

```json
{
  "answer": "Không quá 10 ngày kể từ ngày nhận đủ hồ sơ...",
  "reasoning": "Căn cứ Điều 29..."
}
```

---

## Action plan

```
1. Split raw_unified.jsonl thành train/val/test theo task_type.
2. Inspect và sửa 2 label bất thường (label=7, label=8) trong multi_choice.
3. Đổi prompt: đưa gold answer vào, bắt sinh reasoning JSON.
4. Chạy thử 100 samples.
5. Report parse_success, answer_match_gold, reasoning_length.
6. Chỉ giữ sample clean để train.
7. Dùng CMC-Legal-32B cho subset hoặc làm judge.
8. Train Qwen3-4B trên distilled_train_clean.jsonl.
9. Evaluate trên test.jsonl nguyên bản.
```
