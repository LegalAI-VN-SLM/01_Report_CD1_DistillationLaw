# Script Trình Bày — Pipeline Chưng Cất Dữ Liệu Pháp Luật Việt Nam

> Dùng để trình bày với thầy. Giọng tự nhiên, kể chuyện có đầu có đuôi.

---

## Mở đầu — Đặt vấn đề

Dạ thưa thầy, mục tiêu của em là xây dựng một mô hình ngôn ngữ nhỏ gọn — cỡ 4 tỷ tham số — có khả năng **suy luận pháp lý tiếng Việt**. Tức là không chỉ trả lời đúng, mà còn phải giải thích được *vì sao* đáp án đó đúng, căn cứ vào điều luật nào.

Vấn đề là: dữ liệu pháp luật tiếng Việt hiện tại chỉ có câu hỏi và đáp án — **thiếu hoàn toàn phần lập luận**. Không có "sách giáo khoa" nào dạy model cách suy luận pháp lý cả.

Vì vậy em dùng phương pháp **Knowledge Distillation**: lấy một model lớn (Teacher) sinh ra lời giải thích chi tiết, rồi dùng chính dữ liệu đó để dạy lại model nhỏ (Student).

---

## Phần 1 — Dữ liệu gốc

Em bắt đầu từ bộ dữ liệu **ViLawQA** trong cuộc thi VLSP 2023, tổng cộng **3.428 câu hỏi** chia làm 3 dạng:

- **1.007 câu trắc nghiệm** (multi-choice A/B/C/D) — kiểu "Ai có quyền thừa kế theo pháp luật?"
- **1.421 câu NLI** — cho một điều luật và một câu hỏi, xác định điều luật đó có liên quan không
- **1.000 câu syllogism** — trả lời câu hỏi pháp lý kèm căn cứ điều luật cụ thể

Ba nguồn này đến từ 3 file Parquet khác nhau, em viết script convert và chuẩn hóa tất cả về **một schema JSON thống nhất**, gọi là `raw_unified.jsonl`.

Ngay bước này em đã phát hiện **2 record bất thường** trong multi-choice: có label = 7 và label = 8, trong khi MCQ chỉ có 4 đáp án (0–3). Em đánh dấu để sửa trước khi đưa vào pipeline.

---

## Phần 2 — Thiết kế Pipeline

Pipeline distillation của em gồm **4 module** tách rõ ràng:

1. **Loader** — tải model và tokenizer vào GPU
2. **Generator** — đóng gói prompt theo định dạng ChatML, chạy batch inference
3. **Parser** — trích xuất JSON từ output của Teacher, áp dụng các quality gate để kiểm tra chất lượng
4. **Pipeline** — điều phối toàn bộ, lưu kết quả thành 2 file: clean (dùng để train) và rejected (dùng để debug)

Một nguyên tắc rất quan trọng mà em tuân thủ nghiêm ngặt: **chia data trước, distill sau**. Em split 80/10/10 theo task_type trước khi cho Teacher nhìn bất kỳ dòng nào. Phần test 10% tuyệt đối không bao giờ đi qua Teacher — nếu không sẽ bị data leakage và accuracy sẽ ảo.

---

## Phần 3 — Thử nghiệm đầu tiên và thất bại

Em chọn **SeaLLMs-v3-7B-Chat** làm Teacher đầu tiên vì nó hỗ trợ tiếng Việt tốt. Chạy thử 20 mẫu multi-choice...

Kết quả khá tệ ạ:

- Parse thành công: 80%
- Accuracy tổng thể: chỉ **40%**
- Answer format sạch (đúng A/B/C/D): chỉ **25%**

Model sinh ra đủ kiểu format: lúc trả "C", lúc trả "The exact answer text is: D...", có mẫu còn trả "E" trong khi bài chỉ có 4 đáp án. Có mẫu reasoning viết "B đúng" nhưng answer lại ghi "A" — mâu thuẫn nội tại.

Rõ ràng với chất lượng này, nếu em train Student thì Student sẽ **học nhiễu**: answer sai, format lộn xộn, reasoning yếu.

---

## Phần 4 — Mổ xẻ nguyên nhân gốc

Em dừng lại, không vội sửa model hay đổi Teacher, mà **human review từng sample** để tìm nguyên nhân gốc. Và phát hiện quan trọng nhất là:

> **Vấn đề không nằm ở Teacher — nằm ở data schema và cách em đặt bài toán cho nó.**

Cụ thể có 3 root causes:

**Thứ nhất — NLI schema sai từ bước convert.** File converter của em hard-code `label_semantics` sai và sinh ra choices bị duplicate. Ví dụ 3 trên 4 lựa chọn giống hệt nhau. Nếu để nguyên, Student sẽ học shortcut: "thấy 3 cái giống nhau thì chọn cái khác" — thay vì hiểu ngữ nghĩa thật sự.

**Thứ hai — Reasoning tiếng Anh.** SeaLLMs là model đa ngôn ngữ, khi prompt không đủ chặt, nó tự chuyển sang tiếng Anh. Ví dụ: "The legal document provided is directly related..." — hoàn toàn không phù hợp cho legal SLM tiếng Việt.

**Thứ ba — Syllogism objective sai.** Prompt cũ của em hỏi Teacher: "câu này có trả lời được không?" — nên Teacher trả "answerable". Trong khi em cần nó trả lời: "Không quá 10 ngày kể từ ngày nhận đủ hồ sơ; nếu cần xác minh thì tối đa 45 ngày." Một bên là classification, một bên là QA thật sự.

Em cũng nhận ra rằng `teacher_valid = true` trong parser chỉ có nghĩa là **parse được JSON** — không đồng nghĩa với sample có chất lượng tốt. Cần thêm nhiều lớp kiểm tra hơn.

---

## Phần 5 — Bước ngoặt: Đổi chiến lược

Từ phân tích trên, em quyết định **thay đổi chiến lược từ gốc rễ**:

**Trước đây:** Đưa câu hỏi + các lựa chọn → Teacher tự chọn đáp án và sinh reasoning.
**Bây giờ:** Đưa câu hỏi + các lựa chọn + **đáp án đúng** → Teacher chỉ cần sinh reasoning.

Logic rất đơn giản: SeaLLMs-7B không đủ tin cậy để tự chọn đáp án (accuracy 40%). Nhưng nó **vẫn rất giỏi viết giải thích tiếng Việt nếu đã biết đáp án đúng**. Giống như: một sinh viên có thể không tự tìm ra đáp án, nhưng nếu cho đáp án rồi thì viết bài giải thích rất tốt.

Em cũng sửa toàn bộ prompt:
- Ép JSON output nghiêm ngặt
- Bắt buộc tiếng Việt
- Với Syllogism: đổi từ "có trả lời được không" sang "hãy viết lại lời giải pháp lý, giữ đủ con số quan trọng"
- Với NLI: đổi từ MCQ 4 lựa chọn sang binary relevant/irrelevant sạch

Về đội hình Teacher, em xác định:
- **CMC-AI-Legal-32B** làm Teacher chính (accuracy cao nhất cho legal domain)
- **SeaLLMs-v3-7B** làm Teacher phụ sinh reasoning
- **Qwen2.5-7B** backup

---

## Phần 6 — Hệ thống Quality Gates

Em xây dựng **9 quality gates** trong parser.py, mỗi gate bắt một loại lỗi cụ thể:

1. **Gate 1–2:** Kiểm tra schema đầu vào và parse JSON
2. **Gate 3–4:** Answer hoặc reasoning rỗng
3. **Gate 5:** Reasoning quá ngắn (< 20–40 ký tự tùy task)
4. **Gate 6:** Reasoning tiếng Anh (> 30% từ tiếng Anh)
5. **Gate 7:** Answer không khớp gold label
6. **Gate 8:** Syllogism answer chỉ là "answerable" thay vì nội dung pháp lý
7. **Gate 9:** Syllogism reasoning thiếu con số/thời hạn quan trọng (overlap key terms < 15%)

Mỗi record phải vượt qua **tất cả** 9 gates. Bất kỳ gate nào reject thì record đi vào file rejected, kèm `teacher_error` ghi rõ lý do — để em biết chính xác cần sửa gì.

---

## Phần 7 — Ngưỡng Go/No-Go và lộ trình

Trước khi train Student, em đặt ra **ngưỡng cứng**:

| Metric | Yêu cầu |
|---|---|
| Parse success rate | ≥ 95% |
| Answer match gold | ≥ 98% |
| Empty reasoning | ≤ 2% |
| Invalid answer | ≤ 1% |

Cộng thêm manual review 50 records: reasoning phải đúng pháp lý, tiếng Việt, đủ facts.

Nếu chưa đạt → sửa prompt/parser, chưa train Student. Em không muốn lãng phí compute cho data chưa sạch.

Lộ trình gồm 5 phase:
- **Phase 0:** Split data
- **Phase 1:** Pilot 100–300 samples, check metrics
- **Phase 2:** Distill full train set
- **Phase 3:** Student SFT (Qwen3-4B + QLoRA, BF16, 2–3 epochs)
- **Phase 4:** Evaluate trên test set gốc
- **Phase 5:** Production hardening

---

## Kết — Tổng kết

Tóm lại, hành trình này em rút ra được mấy bài học quan trọng:

1. **Data quality quan trọng hơn model size.** Vấn đề không phải SeaLLMs yếu — mà là em đặt bài toán sai cho nó.

2. **Đừng vội train.** 20 mẫu debug đã tiết lộ đủ thông tin để em tránh lãng phí hàng giờ compute trên data bẩn.

3. **Pipeline cần nhiều lớp bảo vệ.** Từ schema validation, parse check, đến semantic quality gates — mỗi lớp bắt một loại lỗi mà lớp trước bỏ sót.

4. **Gold-conditioned reasoning** là chiến lược phù hợp khi Teacher không đủ mạnh để tự chọn đáp án nhưng vẫn viết giải thích tốt.

Hiện tại em đã có pipeline hoàn chỉnh, prompt đã được redesign, quality gates đã sẵn sàng. Bước tiếp theo là chạy pilot 100 samples với prompt mới và đo metrics trước khi scale lên toàn bộ train set.

---

*Hết.*
