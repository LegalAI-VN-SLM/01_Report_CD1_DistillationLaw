# KẾ HOẠCH THỰC HIỆN THEO TUẦN - HK2/2025-2026

## 📌 THÔNG TIN ĐỀ TÀI & HỌC VIÊN
* **Tên đề tài (English):** *A Hybrid Knowledge Distillation and Preference Optimization Framework for Vietnamese Legal Reasoning in Small Language Models*
* **Tên đề tài (Vietnamese):** *Khung tích hợp Chưng cất tri thức và Tối ưu hóa sở thích cho tác vụ Suy luận pháp lý Việt Nam trên các Mô hình ngôn ngữ nhỏ*
* **Học viên thực hiện:** Hoàng Đình Quý Vũ
  * **MSSV:** 252805008
  * **Email:** hoangdinhquyvu.snape.22@gmail.com
  * **SĐT:** 0868245465
* **Giảng viên hướng dẫn (GVHD):** PGS. TS. Lê Anh Cường
  * **Email:** leanhcuong@tdtu.edu.vn
* **Thời gian thực hiện đề tài:** 16/03/2026 - 05/07/2026 (16 tuần)

### 📅 CÁC CỘT MỐC QUAN TRỌNG
| STT | Nội dung công việc | Thời gian thực hiện |
| :--- | :--- | :--- |
| 1 | Báo cáo giữa kỳ | 13/04/2026 - 19/04/2026 |
| 2 | Kiểm tra đạo văn bài làm | 13/06/2026 - 18/06/2026 |
| 3 | Nộp báo cáo có xác nhận của GVHD | 16/06/2026 - 23/06/2026 |
| 4 | Bộ môn phân công hội đồng phản biện | 16/06/2026 - 21/06/2026 |
| 5 | Báo cáo hội đồng | 23/06/2026 - 30/06/2026 |
| 6 | Nộp điểm chuyên đề nghiên cứu | 01/07/2026 - 03/07/2026 |

---

# Tuần 1
- Từ: 16/3/2026
- Đến: 22/3/2026
- Công việc cần làm: 
  - Đề xuất đề tài và xin phê duyệt đề cương nghiên cứu chi tiết: Lựa chọn tên đề tài mới liên quan đến tác vụ suy luận pháp lý Việt Nam trên các mô hình ngôn ngữ nhỏ (SLMs).
  - Tìm kiếm và đọc các bài báo nghiên cứu khoa học liên quan đến chưng cất tri thức (Knowledge Distillation - KD), tối ưu hóa sở thích (DPO), và suy luận logic pháp lý.
  - Tóm tắt, làm quen các khái niệm và khảo sát các tập dữ liệu để chuẩn bị cho các tuần tiếp theo.
  - Thiết lập môi trường lập trình và thử nghiệm ban đầu (Python, PyTorch, Hugging Face, Unsloth).
- Công việc đã làm: 
  - ✅ GVHD Lê Anh Cường phê duyệt đề tài và thống nhất hướng nghiên cứu.
  - ✅ Đọc hiểu các paper nền tảng như LoRA (Hu et al.), DPO (Rafailov et al.), và Self-Distilled Reasoner.
  - ✅ Khởi tạo môi trường ảo Python và cài đặt các thư viện cần thiết.
- % hoàn thành: 100% 

---

# Tuần 2
- Từ: 23/3/2026
- Đến: 29/3/2026
- Công việc cần làm: 
  - Tải tập dữ liệu huấn luyện `ViLawQA` (2.603 mẫu huấn luyện, 322 mẫu validation, 325 mẫu Split-Test).
  - Thu thập tập dữ liệu `VLSP2025 Public-Test` (440 mẫu) làm benchmark đánh giá ngoài miền phân phối (out-of-distribution - OOD).
  - Tiền xử lý dữ liệu: Làm sạch văn bản, trích xuất căn cứ pháp lý và câu hỏi, chuẩn hóa cấu trúc dạng JSON và format HuggingFace.
  - Tải và cấu hình các mô hình Student nền tảng: `Qwen3-0.6B` và `Qwen3-1.7B`.
  - Chạy Baseline SFT (Supervised Fine-Tuning) gốc trên cả 3 tác vụ: MCQ (Trắc nghiệm), NLI (Nhận định Luật), và Syllogism (Tam đoạn luận) để lấy điểm gốc (Accuracy, F1, ROUGE-L).
- Công việc đã làm: 
  - ✅ Chuẩn hóa thành công toàn bộ tập dữ liệu huấn luyện và kiểm thử sang định dạng ViLawQA.
  - ✅ Cài đặt và chạy baseline thành công cho cả hai mô hình Student 0.6B và 1.7B trên Kaggle GPU.
  - ✅ Đánh giá baseline và ghi nhận kết quả: SFT 0.6B đạt Overall 0.4282, SFT 1.7B đạt Overall 0.4840.
- % hoàn thành: 100% 

---

# Tuần 3
- Từ: 30/3/2026
- Đến: 5/4/2026
- Công việc cần làm: 
  - Thiết lập kết nối với mô hình Teacher lớn `Qwen3-4B-legal-GRPO` (tải định dạng lượng tử hóa 4-bit thông qua bitsandbytes/Unsloth).
  - Viết script prompt mồi để sinh câu trả lời và log-probabilities (nhãn mềm/soft labels) từ mô hình Teacher trên tập `ViLawQA`.
  - Chạy inference mô hình Teacher để kết xuất logits mềm (Top-50 logits tháo) cho 2.603 mẫu huấn luyện làm dữ liệu chưng cất.
  - Lọc và kiểm tra thủ công 50-100 mẫu để đảm bảo không lỗi định dạng hoặc lỗi logic pháp luật.
- Công việc đã làm: 
  - ✅ Nạp thành công mô hình Teacher 4B bằng FastLanguageModel của Unsloth để tránh lỗi tràn VRAM.
  - ✅ Inference thành công và lưu trữ toàn bộ logits mềm của Teacher cho 2.603 mẫu huấn luyện.
  - ✅ Ghi nhận dữ liệu logits đạt chất lượng cao, súc tích và có định dạng `<think>` chuẩn pháp lý.
- % hoàn thành: 100% 

---

# Tuần 4
- Từ: 6/4/2026
- Đến: 12/4/2026
- Công việc cần làm: 
  - Cấu hình các tham số adapter LoRA và QLoRA bằng thư viện PEFT cho mô hình Student.
  - Chạy thử nghiệm chưng cất logit ngoại tuyến (Offline KD V1) với cấu hình thử nghiệm (Learning Rate $1.0 \times 10^{-4}$, effective batch size 16, $\alpha = 0.5$, $T=2.0$).
  - Theo dõi đồ thị loss và phát hiện, phân tích hiện tượng quên thảm họa (catastrophic forgetting) của mô hình.
- Công việc đã làm: 
  - ✅ Hoàn thành cấu hình PEFT (rank $r=16$, $\alpha=32$, target modules ở mức gate/down/up projections).
  - ✅ Phát hiện CE loss của V1 tăng vọt từ $0.3 \rightarrow 0.7$, dẫn đến mô hình sinh lỗi lặp từ và code-switching Anh-Việt-Trung.
  - ✅ Đề xuất cấu hình sửa lỗi V2: giảm LR xuống $2.0 \times 10^{-5}$, nâng effective batch size lên 64, tăng trọng số hard loss ($\alpha=0.3$).
- % hoàn thành: 100% 

---

# Tuần 5
- Từ: 13/4/2026
- Đến: 19/4/2026
- **📋 Báo cáo giữa kỳ: 13/04 - 19/04/2026**
- Công việc cần làm: 
  - Tổng hợp dữ liệu thực nghiệm ban đầu và kết quả SFT baseline viết báo cáo giữa kỳ.
  - Gửi báo cáo tiến độ và xin phê duyệt tiếp tục triển khai từ GVHD Lê Anh Cường.
  - Lập trình viết mã nguồn thuật toán chưng cất tri thức ngoại tuyến (Phase 1: Offline Logit Distillation) với hàm loss joint $\mathcal{L} = \alpha \mathcal{L}_{KD} + (1-\alpha)\mathcal{L}_{CE}$.
  - Thiết lập cơ chế cắt tỉa mạng nơ-ron (Pruning) sử dụng thuật toán Wanda làm nhẹ mô hình Student.
  - Chuẩn bị môi trường tính toán hiệu năng cao (Kaggle Multi-GPU và GPU RTX 6000 Ada).
- Công việc đã làm: 
  - ✅ Hoàn thành báo cáo giữa kỳ và được GVHD thông qua.
  - ✅ Cài đặt xong class huấn luyện `DistillTrainer` hỗ trợ tính toán loss logits song song.
  - ✅ Triển khai thành công thuật toán cắt tỉa Wanda (Layer-wise Pruning) để làm gọn tham số của mô hình học sinh.
- % hoàn thành: 100% 

---

# Tuần 6
- Từ: 20/4/2026
- Đến: 26/4/2026
- Công việc cần làm: 
  - Khởi chạy tiến trình huấn luyện chưng cất Phase 1 (Offline KD) với cấu hình V2 tối ưu trên GPU.
  - Theo dõi đồ thị CE loss và so sánh hành vi huấn luyện giữa LoRA thuần (A1) và QLoRA (A2) trên mô hình 0.6B.
  - Đánh giá hiệu năng của các checkpoint Offline KD thu được trên tập Split-Test.
- Công việc đã làm: 
  - ✅ Huấn luyện thành công mô hình chưng cất Phase 1 cho cả hai scale 0.6B và 1.7B.
  - ✅ Chỉ ra vai trò điều chuẩn ẩn của QLoRA: CE loss giảm ổn định về 0.37 so với mức 0.56 của LoRA thuần, khắc phục hoàn toàn hiện tượng quên thảm họa.
  - ✅ Checkpoint chưng cất 1.7B cải thiện Overall từ 0.4840 lên 0.5322, nâng gấp đôi điểm Syllogism ROUGE-L lên 0.4208.
- % hoàn thành: 100% 

---

# Tuần 7
- Từ: 27/4/2026
- Đến: 3/5/2026
- Công việc cần làm: 
  - Merge trọng số LoRA/QLoRA Adapter vào mô hình Student gốc sau khi hoàn thành Phase 1.
  - Lượng tử hóa mô hình về định dạng 4-bit (giảm size xuống ~2GB) sử dụng thuật toán AWQ/GPTQ.
  - Test lỗi suy giảm trí tuệ và đo lường Latency chạy thử trên CPU/Laptop cá nhân kiểm tra tốc độ sinh chữ.
  - Lập trình mã nguồn huấn luyện tương tác Phase 2 (On-Policy KD) nhằm khắc phục hạn chế exposure bias.
  - Thiết lập cơ chế sinh rollout tự do của Student ($T=0.7$, $p=0.9$) và kết nối Oracle Teacher.
- Công việc đã làm: 
  - ✅ Thực hiện nén và lượng tử hóa mô hình thành công, kích thước giảm về mức ~2GB chạy mượt mà trên phần cứng cá nhân.
  - ✅ Hoàn thành xây dựng cấu trúc mã nguồn on-policy loop, xử lý lỗi lệch pha logits và load Teacher 4-bit qua Unsloth.
  - ✅ Thiết lập hàm loss KL clipping cấp độ token ở ngưỡng $C=5.0$ để tránh XML token thống trị gradient.
- % hoàn thành: 100% 

---

# Tuần 8
- Từ: 4/5/2026
- Đến: 10/5/2026
- Công việc cần làm: 
  - Chạy đánh giá tự động mô hình Phase 1 trên tập Split-Test và tập out-of-distribution `VLSP2025 Public-Test` (440 mẫu).
  - Khởi chạy huấn luyện Phase 2 (On-Policy KD) trên hệ thống GPU song song (Model Split trên Kaggle 2xT4 cho 0.6B và RTX 6000 Ada cho 1.7B).
  - Nghiên cứu cơ chế căn chỉnh sở thích sử dụng chẩn đoán lỗi (Phase 3: Diagnosis-Driven DPO).
  - Viết script gọi API GPT-4o-mini đóng vai trò LLM Judge để kiểm toán ngữ nghĩa 2.603 rollouts huấn luyện của Student 0.6B.
- Công việc đã làm: 
  - ✅ Trích xuất thành công biểu đồ so sánh Baseline vs Offline KD vs Teacher.
  - ✅ Huấn luyện thành công các phiên bản On-Policy SLM: `A3_onpolicy` đạt điểm cải tiến vượt bậc (Overall 0.5075, NLI Accuracy tăng lên 0.7870), trong khi `A2_onpolicy` sụt giảm điểm do tham số quá bé.
  - ✅ Hoàn thành viết mã nguồn kiểm toán ngữ nghĩa và đối chứng chẩn đoán lỗi.
- % hoàn thành: 100% 

---

# Tuần 9
- Từ: 11/5/2026
- Đến: 17/5/2026
- Công việc cần làm: 
  - Xây dựng tập dữ liệu so sánh sở thích (Preference Dataset) thông qua lấy mẫu phân tầng từ kết quả LLM Judge.
  - Thực hiện huấn luyện căn chỉnh DPO (Phase 3) với hệ số $\beta=0.1$, LR $5.0 \times 10^{-6}$, 2 epochs.
- Công việc đã làm: 
  - ✅ LLM Judge "minh oan" tăng số lượng mẫu OK thực tế từ 290 lên 538, phát hiện chính xác các lỗi suy luận sâu giúp tăng số mẫu WRONG từ 1.054 lên 1.652.
  - ✅ Huấn luyện thành công mô hình DPO SLM: `A3_dpo` đạt điểm cao nhất hệ thống (Overall 0.5280, NLI Acc 81.56%, Syllogism ROUGE-L 0.5684).
- % hoàn thành: 100%

---

# Tuần 10
- Từ: 18/5/2026
- Đến: 24/5/2026
- Công việc cần làm: 
  - Viết nháp báo cáo chuyên đề Chương 1 & 2: Mở đầu & Đặt vấn đề, Cơ sở lý thuyết & Nghiên cứu liên quan (KD, DPO, SLMs, đặc thù luật).
  - Viết nháp Chương 3: Phương pháp thực hiện (Kiến trúc đề xuất 3-phase, Wanda pruning, AWQ quantization).
  - Chuẩn hóa danh mục tài liệu tham khảo theo quy định của trường (chuẩn APA).
- Công việc đã làm: 
  - ✅ Viết nháp hoàn chỉnh Chương 1, Chương 2, Chương 3 đạt chuẩn cấu trúc báo cáo khoa học.
  - ✅ Tổng hợp số liệu thực nghiệm và vẽ 6 biểu đồ PNG phục vụ báo cáo.
- % hoàn thành: 100% 

---

# Tuần 11
- Từ: 25/5/2026
- Đến: 31/5/2026
- Công việc cần làm: 
  - Viết Chương 4 (Thực nghiệm & Kết quả) và Chương 5 (Kết luận & Hướng phát triển).
  - Viết file `README.md` hướng dẫn cài đặt code chi tiết cho GVHD.
  - Thiết kế Slide thuyết trình (tập trung vào kiến trúc 3-phase và kết quả thực nghiệm).
  - Quay video demo hệ thống (3-5 phút) upload Drive/YouTube.
- Công việc đã làm: 
  - ✅ Hoàn thiện toàn bộ các chương báo cáo, căn chỉnh các bảng biểu LaTeX (in đậm Top-1, gạch chân Top-2, thêm đường kẻ dọc phân cách Split-Test vs VLSP2025).
  - ✅ Viết tài liệu `README.md` rõ ràng các bước cài đặt thư viện và chạy suy luận.
  - ⏳ Slide thuyết trình và video demo hệ thống chưa hoàn thành.
- % hoàn thành: 70% 

---

# Tuần 12
- Từ: 1/6/2026
- Đến: 7/6/2026
- Công việc cần làm: 
  - Trình GVHD Lê Anh Cường ký duyệt bản báo cáo chuyên đề và nộp đợt 1 lên cổng thông tin khoa/trường.
  - Bộ môn phân công sắp xếp hội đồng phản biện.
  - Tiếp thu ý kiến đóng góp bước đầu từ GVHD để chuẩn bị chỉnh sửa.
- Công việc đã làm: 
  - ✅ GVHD Lê Anh Cường đồng ý thông qua bản báo cáo chuyên đề và ký xác nhận.
  - ✅ Nộp báo cáo đợt 1 thành công lên hệ thống phòng Sau đại học.
- % hoàn thành: 100%

---

# Tuần 13
- Từ: 8/6/2026
- Đến: 14/6/2026
- **⏳ Kiểm tra đạo văn: 13/06 - 18/06/2026**
- Công việc cần làm: 
  - Theo dõi kết quả quét Turnitin đợt 1 từ phòng Sau đại học.
  - Rà soát các cụm từ trùng lặp lớn trong báo cáo, thực hiện paraphrasing để tối ưu hóa tỷ lệ đạo văn.
  - Tinh chỉnh các lỗi định dạng nhỏ trong file LaTeX.
- Công việc đã làm: 
  - ✅ Kết quả đạo văn Turnitin đạt yêu cầu chất lượng cao (<20%).
  - ✅ Sửa đổi các lỗi chính tả và rà soát công thức toán học trong tài liệu.
- % hoàn thành: 100%

---

# Tuần 14
- Từ: 15/6/2026
- Đến: 21/6/2026
- **⏳ Kiểm tra đạo văn: 13/06 - 18/06/2026** (hoàn thành)
- **📝 Nộp báo cáo: 16/06 - 23/06/2026** (đang diễn ra)
- **👥 Phân công hội đồng: 16/06 - 21/06/2026**
- Công việc cần làm: 
  - Nhận phản hồi từ hội đồng phản biện và rà soát các điều chỉnh kỹ thuật cần thiết.
  - Chỉnh sửa bảng DPO ablation (Table 4.8) và Subsection 4.4.2 để bao gồm lớp `PARTIAL` nhằm phản ánh chính xác dữ liệu chẩn đoán.
  - Biên dịch và xuất bản file PDF báo cáo cuối cùng không còn lỗi cảnh báo LaTeX.
  - Nộp file báo cáo hoàn chỉnh có chữ ký của GVHD lên hệ thống.
- Công việc đã làm: 
  - ✅ Hoàn thành sửa đổi Table 4.8 và các phần mô tả tương ứng trong file LaTeX.
  - ⏳ Đang chờ GVHD ký duyệt chính thức và nộp lên hệ thống.
- % hoàn thành: 60%

---

# Tuần 15
- Từ: 22/6/2026
- Đến: 28/6/2026
- **📊 Báo cáo hội đồng: 23/06 - 30/06/2026** (chưa diễn ra)
- Công việc cần làm: 
  - Cập nhật và đồng bộ kịch bản thuyết trình (`presentation_script.md`) khớp với các chỉ số và thuật ngữ mới của mô hình (ROUGE-1, F1-Score, 4 nhóm chẩn đoán).
  - Hoàn thiện slide thuyết trình và luyện tập theo khung thời gian 10 phút nói.
  - Chuẩn bị phương án trả lời câu hỏi phản biện từ hội đồng.
- Công việc đã làm: 
  - ⏳ Chưa diễn ra.
- % hoàn thành: 0%

---

# Tuần 16
- Từ: 29/6/2026
- Đến: 5/7/2026
- **📊 Báo cáo hội đồng: 23/06 - 30/06/2026** (chưa diễn ra)
- **⏳ Nộp điểm: 01/07 - 03/07/2026**
- Công việc cần làm: 
  - Thực hiện báo cáo và bảo vệ đồ án Capstone Project 1 chính thức trước Hội đồng đánh giá chuyên đề khoa.
  - Giải trình các thắc mắc của hội đồng về phương pháp chưng cất, tối ưu hóa sở thích và quantization.
  - Hoàn tất các thủ tục hành chính nộp điểm chuyên đề nghiên cứu về trường.
- Công việc đã làm: 
  - ⏳ Chưa diễn ra.
- % hoàn thành: 0%
