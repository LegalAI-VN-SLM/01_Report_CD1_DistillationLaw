# Evaluation — 3 Task Types

Dataset: **vilawqa** — Vietnamese Legal QA  
Test set: `data/processed/sft/test/`

---

## Tổng quan 3 task

| Task | File test | n | Dạng bài | Metric chính |
|------|-----------|---|----------|-------------|
| NLI | `vilawqa_nli_sft_test.jsonl` | 188 | Binary classification | Accuracy, F1 |
| Multi-choice | `vilawqa_mc_sft_test.jsonl` | 133 | 4-choice (A/B/C/D) | Accuracy |
| Syllogism | `vilawqa_syllo_sft_test.jsonl` | 112 | Open-ended | ROUGE-L, EM |

---

## Task 1 — NLI (Natural Language Inference)

### Bài toán

Cho một đoạn văn bản pháp luật và một câu hỏi, model phán đoán đoạn văn đó có **liên quan** đến câu hỏi hay không.

### Input mẫu

```
### Instruction:
Đánh giá văn bản pháp luật có liên quan đến câu hỏi hay không.

### Input:
### LEGAL DOCUMENT:
Khoản 1 Điều 30 Luật Hải quan 2014...
Tờ khai hải quan điện tử được đăng ký theo phương thức điện tử.

### QUESTION:
Với câu hỏi sau:
Có mấy phương thức đăng ký tờ khai hải quan hiện nay?
Điều luật được cung cấp có liên quan đến câu hỏi hay không?

### Response:
```

### Output expected

Chỉ có 2 nhãn duy nhất:
```
Có liên quan.
Không liên quan.
```

Model có thể có phần `<think>...</think>` trước nhãn — phần này bị strip khi eval.

### Cách extract nhãn từ output model

```python
import re

def extract_nli_label(output: str) -> str:
    # Strip thinking block nếu có
    clean = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()
    if "Có liên quan" in clean:
        return "relevant"
    if "Không liên quan" in clean:
        return "not_relevant"
    return "unknown"
```

### Metrics

- **Accuracy** = số dự đoán đúng / tổng
- **F1** (macro) = trung bình F1 của 2 class

> Baseline ngây thơ (predict "Có liên quan" hết): ~55–60%. Model tốt cần > 85%.

---

## Task 2 — Multi-choice (MC)

### Bài toán

Chọn 1 trong 4 đáp án (A/B/C/D) cho câu hỏi pháp luật.

### Input mẫu

```
### Instruction:
Chọn đáp án đúng nhất cho câu hỏi pháp luật.

### Input:
Câu hỏi: Loại cổ phần nào được chuyển nhượng tự do?

A. Cổ phần phổ thông, các loại cổ phần ưu đãi
B. Cổ phần phổ thông, các loại cổ phần ưu đãi trừ cổ phần ưu đãi biểu quyết
C. Cổ phần phổ thông, các loại cổ phần ưu đãi trừ cổ phần ưu đãi cổ tức
D. Cổ phần phổ thông, các loại cổ phần ưu đãi trừ cổ phần ưu đãi hoàn lại

### Response:
```

### Output expected

```
Đáp án: B. Cổ phần phổ thông, các loại cổ phần ưu đãi trừ cổ phần ưu đãi biểu quyết
```

### Cách extract đáp án

```python
import re

def extract_mc_answer(output: str) -> str | None:
    clean = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()
    # Pattern: ": A." hoặc ": B." ...
    m = re.search(r":\s*([A-D])\.", clean)
    if m:
        return m.group(1)
    # Fallback: tìm chữ cái đứng đầu dòng
    m2 = re.search(r"^([A-D])\.", clean, re.MULTILINE)
    return m2.group(1) if m2 else None
```

### Metrics

- **Accuracy** = số chọn đúng letter / tổng
- Random baseline = 25%. Model tốt cần > 70%.

---

## Task 3 — Syllogism (Open-ended)

### Bài toán

Trả lời câu hỏi pháp luật dựa trên căn cứ pháp lý được cung cấp (hoặc từ kiến thức model). Đây là bài toán **open-ended generation** — không có đáp án cố định.

### Input mẫu

```
### Instruction:
Trả lời câu hỏi pháp luật dựa trên căn cứ được cung cấp.

### Input:
Căn cứ pháp lý:
Điều 19 Luật thuế thu nhập cá nhân 2007...
Mức giảm trừ đối với đối tượng nộp thuế là 4 triệu đồng/tháng.

Câu hỏi:
Mức giảm trừ gia cảnh năm 2024 là bao nhiêu?

### Response:
```

### Output expected

```
Mức giảm trừ gia cảnh năm 2024 là 11 triệu đồng/tháng (132 triệu đồng/năm).
Đối với mỗi người phụ thuộc, mức giảm trừ là 4,4 triệu đồng/tháng.
```

### Metrics

| Metric | Ý nghĩa | Threshold tốt |
|--------|---------|--------------|
| **ROUGE-L** | Overlap chuỗi dài nhất với reference | > 0.40 |
| **Exact Match (EM)** | Khớp hoàn toàn (sau normalize) | tham khảo thêm |
| **ROUGE-1/2** | Overlap unigram/bigram | > 0.50 / 0.30 |

> Lưu ý: ROUGE thấp không nhất thiết là sai — model có thể paraphrase đúng. ROUGE cao hơn khi cùng style với reference.

```python
from rouge_score import rouge_scorer

def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for k in scores:
            scores[k].append(result[k].fmeasure)
    return {k: sum(v) / len(v) for k, v in scores.items()}
```

---

## Data có `thinking` flag — tại sao quan trọng

Test set hiện tại:

| Task | thinking=True | thinking=False |
|------|--------------|----------------|
| NLI  | 141 (75%)    | 47 (25%)       |
| MC   | 100 (75%)    | 33 (25%)       |
| Syllo| 84 (75%)     | 28 (25%)       |

`thinking=True` — ground truth có `<think>...</think>` trước câu trả lời (chain-of-thought).  
`thinking=False` — ground truth trả lời thẳng, không có reasoning block.

**Tại sao cần tách?**

- Model có thể học reasoning từ `thinking=True` samples tốt hơn cho các câu hỏi phức tạp.
- Nếu `acc(thinking=True) >> acc(thinking=False)` → model phụ thuộc vào CoT, không tự trả lời thẳng được.
- Nếu `acc(thinking=False) >> acc(thinking=True)` → model bị confuse bởi reasoning dài, cần thêm data CoT.

---

## Qwen3 thinking mode khi inference

Qwen3 có 2 mode inference. Cách kiểm soát bằng prefix inject vào cuối prompt:

| Mode | Prefix thêm vào sau `### Response:\n` | Hành vi |
|------|--------------------------------------|---------|
| **Thinking ON** | `<think>\n` | Model tự viết reasoning rồi đóng `</think>`, sau đó trả lời |
| **Thinking OFF** | `<think>\n\n</think>\n\n` | Model bỏ qua reasoning, trả lời thẳng |

> Đây là technique chính thức của Qwen3, không phải hack — xem [Qwen3 docs](https://huggingface.co/Qwen/Qwen3-1.7B).

Khi eval, `strip_think()` xóa toàn bộ `<think>...</think>` block trước khi tính metrics — chỉ đánh giá phần trả lời cuối.

---

## Chạy eval

### Thinking mode OFF (trả lời thẳng)

```powershell
uv run python scripts/eval.py \
  --config configs/qwen3_1b7_legal.yaml \
  --adapter outputs/legal-slm-sft-v1/adapter_final \
  --test_dir data/processed/sft/test \
  --output_dir outputs/evals/v1-no-think
```

### Thinking mode ON (có chain-of-thought)

```powershell
uv run python scripts/eval.py \
  --config configs/qwen3_1b7_legal.yaml \
  --adapter outputs/legal-slm-sft-v1/adapter_final \
  --test_dir data/processed/sft/test \
  --output_dir outputs/evals/v1-with-think \
  --thinking_mode
```

> **Nên chạy cả 2** để so sánh — biết khi nào nên bật/tắt thinking trong production.

### Output

```
outputs/evals/<run>/
├── nli_results.json      accuracy, f1, split by thinking flag, per-sample
├── mc_results.json       accuracy, split by thinking flag, per-sample
├── syllo_results.json    rouge1/2/L, split by thinking flag, per-sample
└── summary.json          tổng hợp + breakdown thinking=True/False
```

---

## Interpret kết quả

```
NLI accuracy  > 85%   → model hiểu relevance tốt
MC  accuracy  > 70%   → model có kiến thức pháp luật
Syllo ROUGE-L > 0.40  → model generate đúng hướng
```

**Đọc breakdown `by_thinking`:**

```
nli_acc_thinking  > nli_acc_direct  → CoT giúp ích, bật thinking_mode khi deploy
nli_acc_direct    > nli_acc_thinking → model ổn định hơn khi trả lời thẳng
```

Nếu NLI thấp mà MC cao → model biết facts nhưng không phân biệt được relevance.  
Nếu MC thấp mà Syllo ROUGE cao → model paraphrase tốt nhưng reasoning yếu.
