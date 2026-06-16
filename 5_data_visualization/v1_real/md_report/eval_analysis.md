# Evaluation Analysis: Dataset Consistency & Reliability (Phân tích Đánh giá: Tính Thống nhất & Độ Tin cậy của Dữ liệu)

Tài liệu này phân tích chi tiết về phương pháp đánh giá (Evaluation), tập dữ liệu sử dụng, tính ổn định thực tế của các chỉ số (metrics) và các lỗi không thống nhất kỹ thuật giữa các giai đoạn thực nghiệm của dự án.

---

## 1. Dữ liệu Đánh giá (Evaluation Dataset) là gì?
Tất cả các thử nghiệm chưng cất tri thức trong dự án đều thống nhất đánh giá trên tập kiểm thử độc lập:
* **Tên tệp dữ liệu:** `vilawqa_sft_test.jsonl` (Được chia tách độc lập khỏi tập huấn luyện `train` và tập kiểm thử chéo `valid`).
* **Tổng số lượng mẫu:** **325 mẫu**, phân chia theo 3 nhóm tác vụ chính:
  1. **NLI (Nhận định Luật - Có/Không liên quan):** 141 mẫu.
  2. **MCQ (Trắc nghiệm chọn đáp án A/B/C/D):** 100 mẫu.
  3. **Syllogism (Lập luận Logic Tam đoạn luận):** 84 mẫu.
* **Đặc trưng cấu trúc:** Khoảng **$75\%$** số mẫu trong tập test chứa cấu trúc khối suy nghĩ `<think> ... </think>` đi kèm câu trả lời cuối cùng để đánh giá năng lực suy luận chuỗi (reasoning chain).

---

## 2. Kỹ thuật Đánh giá Offline (Rule-based & ROUGE-L) có thực sự ổn định?
Phương pháp đánh giá offline ban đầu sử dụng bộ chấm dựa trên luật (Rule-based) và so khớp ký tự:
* **Tác vụ NLI & MCQ:** Dùng biểu thức chính quy (Regex) để trích xuất từ khóa kết luận đầu tiên (như `"Co lien quan"`, `"Khong lien quan"`, hoặc các ký tự `[A-D]`).
* **Tác vụ Syllogism:** Tách phần văn bản sau thẻ `</think>` và tính điểm tương đồng từ vựng **ROUGE-1** và **ROUGE-L** so với câu trả lời chuẩn (Gold Answer).

### Đánh giá độ tin cậy thực tế: **Cực kỳ thiếu ổn định (Unstable)**
Dù bộ eval này chạy nhanh và không tốn chi phí API, nó bộc lộ 3 nhược điểm lớn làm sai lệch kết quả báo cáo:
1. **Lỗi trích xuất (False Negative):** Regex trích xuất rất nhạy cảm với format. Nếu Student trả lời đúng ngữ nghĩa nhưng diễn đạt khác đi một chút (ví dụ: viết `"Có, tài liệu này liên quan đến..."` thay vì đúng cụm `"Có liên quan"`), Regex sẽ đánh giá là **SAI**. Điều này khiến mô hình bị chấm oan.
2. **Dễ dãi đối với lập luận (False Positive):** Điểm ROUGE-L chỉ đo đạc sự trùng lặp từ vựng bề mặt (overlap). Một câu trả lời sai hoàn toàn về logic pháp luật hoặc trích sai số tiền xử phạt nhưng chứa nhiều từ khóa trùng lặp với luật gốc vẫn được điểm ROUGE-L cao.
3. **Minh chứng đột phá từ Phase 3 (LLM Judge):** Khi dự án chuyển sang dùng **LLM Judge (GPT-4o-mini)** để chấm điểm ngữ nghĩa thực tế trên tập train:
   * Số câu trả lời đúng thực chất (**OK**) tăng vọt từ **11.1% lên 20.7%** (do LLM Judge minh oan cho các câu trả lời đúng ngữ nghĩa nhưng sai format mà Regex bỏ sót).
   * Số câu trả lời sai hoàn toàn (**WRONG**) tăng mạnh từ **40.5% lên 63.5%** (do LLM Judge phát hiện ra các lỗi suy luận logic sai thực chất hoặc trích sai điều luật mà điểm ROUGE-L vô tình bỏ qua).
   
**Kết luận:** Bộ đánh giá offline dựa trên Regex + ROUGE-L chỉ mang tính chất tham khảo tương đối. Để đánh giá chính xác năng lực tư duy pháp lý, bắt buộc phải kết hợp đánh giá ngữ nghĩa thông qua LLM Judge.

---

## 3. Tính Thống nhất của Dữ liệu Đánh giá qua các Runs

### Những điểm ĐÃ THỐNG NHẤT:
* **Tập dữ liệu kiểm thử:** Cố định duy nhất tệp `vilawqa_sft_test.jsonl` (325 mẫu) qua toàn bộ 10 runs huấn luyện.
* **Bộ đo đạc chính:** Điểm **Best Overall** được tính bằng trung bình cộng trọng số của MCQ Accuracy, NLI Accuracy, và Syllogism ROUGE-L.

### Những điểm BẤT NHẤT NGUY HIỂM (Cần lưu ý trong Báo cáo):
1. **Bất nhất về Chiến lược Giải mã (Decoding Strategy) ở chặng V1:**
   * Trong giai đoạn đầu (**KD V1** gồm `E1`, `E2`, `E3`), cấu hình đánh giá bị gán lỗi **`do_sample=True`** với nhiệt độ $T=0.7$. Điều này khiến quá trình giải mã bị ngẫu nhiên hóa, mô hình sinh ra các câu trả lời khác nhau qua mỗi lần chạy thử, làm cho các đường cong eval score bị trồi sụt cực kỳ nhiễu và kết quả Best Overall không thể tái tạo ổn định.
   * Sang chặng **KD V2** (các runs nhóm `A`), lỗi này đã được khắc phục triệt để bằng cách cố định giải mã tham lam **Greedy Decoding (`do_sample=False`, `temperature=1.0` hoặc `0.0`)**, đảm bảo tính nhất quán tuyệt đối của câu trả lời.
2. **Bất nhất về Tên cột ghi nhận logs (Metric Logging Names):**
   * Giữa chặng V1 và V2, tên các cột ghi log điểm số trên Wandb bị thay đổi:
     * *NLI F1:* V1 ghi nhận dưới tên `eval/nli_f1_macro` nhưng V2 ghi nhận dưới tên `eval/nli_f1`.
     * *Syllogism ROUGE-L:* V1 ghi là `eval/syl_rouge_l` nhưng V2 ghi là `eval/syl_rougeL`.
   * Sự bất nhất này khiến cho việc đối chiếu bảng số liệu thô trực tiếp từ Wandb dễ gặp sai sót (script phân tích tự động `extract.py` của dự án đã phải cài đặt hàm chuẩn hóa gộp tên cột để khắc phục lỗi này).
3. **Lỗi ghi nhận logs (Log Logging Bug) của On-Policy KD:**
   * Trong run `A3_onpolicy_onpolicy_kd`, chỉ số `eval/best_overall` bị ghi nhận lỗi bằng $0$ trong tệp `summary.json`. 
   * Phân tích cho thấy mô hình vẫn được đánh giá bình thường và ghi điểm ở cột `eval/overall` (đạt **0.44147**). Báo cáo tự động đã phải trích xuất fallback từ cột `eval/overall` cuối chặng để có số liệu chính xác.
