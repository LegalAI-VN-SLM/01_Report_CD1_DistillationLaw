# Workflow Chưng Cất Dữ Liệu (Teacher Distillation)

Tài liệu này giải thích chi tiết luồng xử lý (end-to-end) của toàn bộ pipeline sinh dữ liệu chưng cất, từ lúc lấy dữ liệu thô cho đến khi mô hình Teacher (Thầy giáo) xuất ra lời giải chi tiết.

---

## 1. Tại sao lại bắt đầu từ `raw_unified.jsonl`?

File `data/processed/jsonl/raw_unified.jsonl` là "điểm khởi đầu" của toàn bộ quá trình. 

*   **Tính thống nhất:** Dữ liệu luật pháp thường đến từ rất nhiều nguồn khác nhau (dataset trắc nghiệm ViLawQA, syllogism, NLI...). File `raw_unified.jsonl` là phiên bản đã được **chuẩn hóa** về một cấu trúc JSON duy nhất (có `id`, `input`, `target`, `label`...).
*   **Chưa có suy luận:** File này hiện tại chỉ chứa **Câu hỏi** và **Đáp án đúng (A, B, C, D)**. Nó hoàn toàn thiếu đi phần giải thích, lập luận pháp lý (Reasoning) tại sao lại chọn đáp án đó.
*   **Vai trò:** Nó đóng vai trò là "Đề bài" để giao cho mô hình Teacher (Thầy giáo lớn như SeaLLMs-7B hay CMC-Legal-32B) giải thích cặn kẽ.

---

## 2. Tại sao phải đưa dữ liệu vào Chat Template?

Mặc dù chúng ta đã có câu hỏi, nhưng nếu chỉ quăng một chuỗi văn bản trần (raw text) vào mô hình ngôn ngữ lớn (LLM), mô hình sẽ rất "bối rối", dẫn đến sinh ra văn bản rác, hoặc tự lặp lại câu hỏi.

Lý do là các mô hình như SeaLLMs hay CMC-Legal đã được huấn luyện (fine-tuned) theo định dạng hội thoại. Chúng cần biết rõ ai đang nói và lúc nào đến lượt mình. Đó là lúc **Chat Template (ví dụ: ChatML)** phát huy tác dụng.

Khi đi qua Chat Template, một câu hỏi đơn giản sẽ được "đóng gói" lại như sau:

```text
<|im_start|>system
Bạn là một chuyên gia pháp luật Việt Nam. Hãy phân tích câu hỏi cẩn thận...<|im_end|>
<|im_start|>user
Câu hỏi: Con bao nhiêu tuổi có thể tự quản lý tài sản?
Các phương án: A... B...
Hãy trả lời theo đúng format:
Giải thích: ...
Đáp án: ...<|im_end|>
<|im_start|>assistant
(Mô hình sẽ tự động hiểu là bắt đầu sinh chữ từ đây)
```

**Lợi ích của Chat Template:**
1.  **Gắn System Prompt:** Ép mô hình đóng vai "Chuyên gia pháp luật", giúp câu trả lời chuyên nghiệp và có căn cứ hơn.
2.  **Kích hoạt Generation:** Token `<|im_start|>assistant` ở cuối cùng là tín hiệu ép mô hình phải bắt đầu trả lời, ngăn chặn việc mô hình bị "hallucination" (ảo giác, nói nhảm).

---

## 3. Toàn cảnh luồng xử lý End-to-End

Dưới đây là chi tiết các bước mà thư viện `slm_data_distillation_law` đang thực hiện khi bạn chạy lệnh pipeline:

### Bước 1: Khởi tạo & Tải mô hình (`loader.py` & `main.py`)
*   Pipeline đọc cấu hình model (từ `models/`).
*   Tải mô hình (Weights) vào VRAM của GPU. Có thể tải qua `Unsloth` (để tăng tốc 2x) hoặc `Transformers` thuần (dùng `bitsandbytes` 4-bit).
*   Đặc biệt: Hệ thống tự động kiểm tra và **vá (patch) Chat Template** nếu Tokenizer của mô hình bị lỗi hoặc thiếu cấu trúc sinh văn bản (`add_generation_prompt`).

### Bước 2: Chuẩn bị dữ liệu (`pipeline.py`)
*   Đọc "đề bài" từ `raw_unified.jsonl`.
*   **Smart Batching (Tối ưu hóa):** Sắp xếp các câu hỏi theo độ dài (số lượng token ước tính). Gom các câu hỏi ngắn với nhau, dài với nhau thành các Batch (ví dụ Batch Size = 5). Điều này giúp GPU xử lý song song mà không lãng phí tài nguyên vào việc "độn" (padding) token rỗng.

### Bước 3: Định dạng Prompt (`prompting.py`)
*   Dịch đổi dữ liệu từ định dạng JSON thô sang chuỗi văn bản trắc nghiệm.
*   Chèn thêm câu lệnh hướng dẫn Chain-of-Thought (CoT): Yêu cầu mô hình phải xuất ra chữ **"Giải thích: ..."** trước, rồi mới chốt **"Đáp án: ..."**.

### Bước 4: Chạy Suy luận - Inference (`generator.py`)
*   Bọc chuỗi văn bản vào **Chat Template** (đã giải thích ở mục 2).
*   Đẩy nguyên Batch vào mô hình Teacher.
*   Mô hình sẽ sinh ra các đoạn văn bản dài chứa quá trình suy luận pháp lý dựa trên câu hỏi. Hệ thống áp dụng `repetition_penalty` để ép mô hình không bị lặp chữ.

### Bước 5: Bóc tách thông tin (`parser.py`)
*   Kết quả trả về từ GPU là một đoạn văn bản (Raw Output).
*   Parser sử dụng Biểu thức chính quy (Regex) để trích xuất:
    *   Cắt lấy phần chữ sau "Giải thích:" gán vào `teacher_reasoning`.
    *   Cắt lấy chữ cái (A, B, C, D) sau "Đáp án:" gán vào `teacher_answer`.
    *   Vẫn giữ lại `teacher_raw_output` để lập trình viên có thể debug nếu Regex cắt sai.

### Bước 6: Lưu kết quả chưng cất (`pipeline.py`)
*   Ghép thông tin Reasoning và Answer vừa lấy được trở lại cùng với ID và Input của câu hỏi ban đầu.
*   Lưu toàn bộ ra file `distilled_teacher.jsonl` (hoặc `distilled_results_v2.jsonl` tuỳ bạn cấu hình).

> **Kết quả:** Từ một file ban đầu chỉ có câu hỏi trắc nghiệm cộc lốc, chúng ta thu được một bộ dữ liệu có đầy đủ lời giải thích cặn kẽ chuẩn văn phong pháp lý. File kết quả này chính là "sách giáo khoa" hoàn hảo để sau này đem đi dạy (Fine-tune/LoRA) cho mô hình Học sinh (Student - SLM) nhỏ gọn hơn.
