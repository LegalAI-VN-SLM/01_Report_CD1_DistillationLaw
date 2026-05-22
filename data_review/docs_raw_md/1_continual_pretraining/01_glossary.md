# 01 — Glossary

Thuật ngữ sử dụng trong dự án Continual Pre-Training cho Legal SLM.

---

## Mô hình & Huấn luyện

**SLM (Small Language Model)**
Mô hình ngôn ngữ cỡ nhỏ (thường < 7B tham số). Dự án này nhắm đến các model như Qwen, Gemma, Llama dạng 1B–3B để deploy hiệu quả.

**Base Model**
Model nền tảng đã được pre-train trên dữ liệu tổng quát, chưa instruction-tuned hay fine-tuned. CPT thực hiện trên base model.

**CPT (Continual Pre-Training)**
Tiếp tục quá trình pre-training của một model đã có trên dữ liệu domain cụ thể (ở đây là pháp luật Việt Nam). Khác với SFT, CPT dùng objective next-token prediction thuần túy, không có instruction/response format.

**SFT (Supervised Fine-Tuning)**
Huấn luyện có giám sát trên dữ liệu dạng instruction–response. Thường thực hiện sau CPT để model học cách trả lời câu hỏi.

**Next-Token Prediction**
Objective huấn luyện của pre-training: với mỗi vị trí trong chuỗi, model học dự đoán token tiếp theo. Loss là cross-entropy trên toàn bộ chuỗi.

**Perplexity (PPL)**
Chỉ số đánh giá language model: PPL thấp = model dự đoán văn bản tốt hơn. Dùng để đo xem CPT có giúp model hiểu legal text tốt hơn không.

**LoRA (Low-Rank Adaptation)**
Kỹ thuật fine-tuning hiệu quả: chỉ train một lượng nhỏ tham số bổ sung, giữ nguyên phần lớn model gốc. Giảm VRAM và thời gian train đáng kể.

**Unsloth**
Thư viện Python tối ưu tốc độ fine-tuning cho LLM, hỗ trợ LoRA + Flash Attention. Dự án SFT hiện tại đang dùng Unsloth.

---

## Dữ liệu

**Parquet**
Định dạng file columnar (theo cột) của Apache, nén tốt và đọc nhanh với pandas/pyarrow. Dataset pháp luật lưu dưới dạng parquet shards.

**Shard**
Một phần của dataset lớn được chia nhỏ thành nhiều file. Dataset gốc có 24 shards (`train-00000-of-00024.parquet` ... `train-00023-of-00024.parquet`). Dự án này dùng 2 shards đầu.

**markitdown**
Thư viện Python của Microsoft (`microsoft/markitdown`) convert nhiều định dạng file (PDF, Word, HTML, Excel, ...) sang Markdown. Trong pipeline v2, dùng để convert `doc_content` từ HTML → Markdown sạch, giữ lại cấu trúc bảng và đề mục thay vì chỉ strip tags. Cài qua `pip install markitdown[all]`.

**`doc_content`**
Column chứa nội dung văn bản pháp luật dạng **HTML** (scrape từ nguồn gốc, chứa `<table>`, `<span style=...>`, `<br/>`, ...). Cần convert sang text/Markdown trước khi dùng cho CPT.

**`metadata`**
Column kiểu dict chứa thông tin mô tả văn bản. Các key:

| Key | Ý nghĩa |
|-----|---------|
| `Id` | ID nội bộ của văn bản |
| `DocIdentity` | Số hiệu văn bản (vd: `12/2021/NĐ-CP`) |
| `DocName` | Tên đầy đủ của văn bản |
| `IssueDate` | Ngày ban hành |
| `OrganName` | Cơ quan ban hành |

**Token / Token Estimate**
Đơn vị đầu vào của LLM sau khi tokenize. Script dùng heuristic: `tokens ≈ words × 1.33` để ước tính nhanh mà không cần chạy tokenizer thực.

**Deduplication (Dedup)**
Loại bỏ các văn bản trùng lặp. Thực hiện theo thứ tự ưu tiên: trùng `Id` → trùng `DocIdentity` → trùng nội dung.

**NFC (Unicode Normalization Form C)**
Chuẩn hóa Unicode dạng composed: ghép các ký tự base + dấu thành một code point duy nhất. Quan trọng với tiếng Việt vì cùng một chữ có thể được encode theo nhiều cách khác nhau.

**Blank Line Ratio**
Tỉ lệ `dòng trống / tổng số dòng` trong một document. Cao bất thường thường là dấu hiệu scrape lỗi hoặc định dạng bị vỡ.

**Non-VN Ratio**
Tỉ lệ ký tự không phải tiếng Việt/Latin trong document. Cao bất thường có thể là encoding error hoặc document sai ngôn ngữ.

---

## Dataset & Nguồn

**VLSP2025-LegalSML/legal-pretrain**
HuggingFace dataset chứa văn bản pháp luật Việt Nam dùng cho pre-training. Nguồn từ cuộc thi VLSP 2025.

**VLSP (Vietnam Language and Speech Processing)**
Hội nghị / cuộc thi NLP tiếng Việt thường niên. VLSP 2025 có task về Legal SLM.

**HuggingFace Hub**
Nền tảng lưu trữ và chia sẻ model, dataset, spaces cho ML. Dataset và model của dự án được host tại đây.

---

## Định dạng Output

**CPT Text Format**
Format của column `text` trong `processed/train.parquet`:
```
{DocName} | {OrganName} | {IssueDate}
{doc_content}
```
Header metadata được prepend để model học được ngữ cảnh của văn bản trong quá trình CPT.
