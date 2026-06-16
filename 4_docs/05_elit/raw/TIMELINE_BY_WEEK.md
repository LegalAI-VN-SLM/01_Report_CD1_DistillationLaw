# KẾ HOẠCH THỰC HIỆN THEO TUẦN - HK2/2025-2026

## 📌 THÔNG TIN ĐỀ TÀI & HỌC VIÊN
* **Tên đề tài (English):** *Mitigating Bias in Vietnamese Legal Small Language Models through Counterfactual Data Augmentation and Self-Imitation Learning*
* **Học viên thực hiện:** Hoàng Đình Quý Vũ
  * **MSSV:** 252805008
  * **Email:** hoangdinhquyvu.snape.22@gmail.com
  * **SĐT:** 0868245465
* **Giảng viên hướng dẫn (GVHD):** Thầy Trần Lương Quốc Đại
  * **Email:** tranluongquocdai@tdtu.edu.vn
* **Thời gian thực hiện đề tài:** 17/03/2026 - 15/06/2026 (2.5 tháng)

### 📅 CÁC CỘT MỐC QUAN TRỌNG
| STT | Nội dung công việc | Thời gian thực hiện |
| :--- | :--- | :--- |
| 1 | Báo cáo giữa kỳ | 13/04/2026 - 19/04/2026 |
| 2 | Kiểm tra đạo văn bài làm | 13/06/2026 - 18/06/2026 |
| 3 | Nộp báo cáo có xác nhận của GVHD | 16/06/2026 - 23/06/2026 |
| 4 | Bộ môn phân công hội đồng phản biện | 16/06/2026 - 21/06/2026 |
| 5 | Báo cáo hội đồng | 23/06/2026 - 30/06/2026 |

---
# Tuần 1
- Từ: 16/3/2026
- Đến: 22/3/2026
- Công việc cần làm: 
  - GVHD phê duyệt đề tài nghiên cứu dựa trên lý do chọn đề tài.
  - Tìm kiếm dữ liệu thô và các nguồn data pháp lý tiếng Việt phù hợp.
  - Tìm kiếm và đọc các bài báo (papers) nghiên cứu liên quan đến định kiến (bias), sự công bằng (fairness) trong mô hình ngôn ngữ lớn (LLMs).
  - Nghiên cứu tập dữ liệu benchmark pháp lý tiếng Việt VLegal-Bench.
  - Thiết lập môi trường lập trình ban đầu (Python, PyTorch, Hugging Face).
- Công việc đã làm: 
  - ✅ GVHD phê duyệt đề tài và định hướng nghiên cứu.
  - ✅ Khảo sát và tìm kiếm các nguồn ngữ liệu pháp lý Việt Nam, xác định VLegal-Bench là tập dữ liệu nền tảng.
  - ✅ Tìm kiếm và đọc 5 bài báo khoa học về đo lường và giảm thiểu bias (như CrowS-Pairs, WEAT).
  - ✅ Phân tích cấu hình và cấu trúc của VLegal-Bench để lọc các task liên quan.
  - ✅ Setup môi trường ảo và cài đặt thư viện cần thiết.
- % hoàn thành: 100% 

---

# Tuần 2
- Từ: 23/3/2026
- Đến: 29/3/2026
- Công việc cần làm: 
  - Tiếp tục tìm kiếm và thu thập dữ liệu về danh xưng và tên đệm tiếng Việt để xây dựng từ điển swap giới tính.
  - Tìm kiếm và đọc các bài báo liên quan đến kỹ thuật Counterfactual Data Augmentation (CDA) và đo lường độ lệch xác suất.
  - Thiết kế khung đánh giá CFRE (Context-Sensitive Fairness and Robustness Evaluation) và vòng lặp tối ưu RL-SIL (Self-Imitation Learning).
  - Thiết kế giải pháp trích xuất Layer 1 Rule-Based Extraction.
  - Viết bộ parser cho các câu trắc nghiệm MCQ (tasks 5.x) để sinh cặp explicit bias.
- Công việc đã làm: 
  - ✅ Thu thập thêm dữ liệu từ vựng danh xưng tiếng Việt để bổ sung vào bộ swap giới tính.
  - ✅ Tìm đọc các bài báo về tối ưu hóa độ ổn định ngữ nghĩa (Representation Consistency Evaluation - RCE) và tính độ lệch phân phối xác suất (FIS) bằng KL Divergence.
  - ✅ Hoàn thành công thức toán học và thiết kế lý thuyết cho chỉ số CFRE (kết hợp FIS dùng Symmetric KL Divergence và RCE dùng Cosine Similarity).
  - ✅ Triển khai mã nguồn parser MCQ (hỗ trợ nhiều định dạng đáp án A:, A., A: text) và bộ swap giới tính một lượt.
- % hoàn thành: 100% 

---

# Tuần 3
- Từ: 30/3/2026
- Đến: 5/4/2026
- Công việc cần làm: 
  - Khảo sát các bộ dữ liệu bias khác trên thế giới và tìm đọc các bài báo về LLM-based data augmentation.
  - Chạy thử nghiệm và thu thập kết quả Layer 1 Rule-Based Extraction trên tập VLegal-Bench.
  - Phân tích chất lượng các cặp dữ liệu đối ngẫu sinh ra ở Layer 1.
  - Xác định các giới hạn kỹ thuật của phương pháp rule-based (lỗi tên riêng không đổi theo giới tính, thiếu nhãn định kiến, thiếu các trục định kiến khác).
  - Lên kế hoạch thiết kế Layer 2 LLM-based Augmentation sử dụng API GPT-4o-mini.
- Công việc đã làm: 
  - ✅ Tìm đọc các bài báo khoa học về sử dụng LLM để sinh dữ liệu phản thực tế có kiểm soát và lọc nhiễu prompt.
  - ✅ Thu được 2,126 cặp đối ngẫu từ Layer 1 (864 từ `5x_direct`, 1,262 từ `gender_swap`).
  - ✅ Phân tích dữ liệu và chỉ ra các lỗi nghiêm trọng về ngữ nghĩa: "Chị Lò Thị V" thành "Anh Lò Thị V" (tên đệm "Thị" bị mâu thuẫn) và 721 cặp bị gắn nhãn `unknown`.
  - ✅ Hoàn thành bản thảo kiến trúc Layer 2 nhằm tự động hóa việc rewrite tên, gán nhãn và sinh 5 trục thuộc tính nhạy cảm mới.
- % hoàn thành: 100% 

---

# Tuần 4
- Từ: 6/4/2026
- Đến: 12/4/2026
- Công việc cần làm: 
  - Lập trình bộ Async client gọi API GPT-4o-mini để xử lý song song khối lượng dữ liệu lớn.
  - Thiết kế và chuẩn hóa 3 bộ prompt templates (Rewrite, Attribute Swap, Classify).
  - Cài đặt cơ chế kiểm soát số lượng requests đồng thời (Semaphore) để tối ưu thời gian chạy.
  - Viết bộ lọc phát hiện lọt hướng dẫn prompt (Prompt Leak Check) trong `validate/checker.py`.
- Công việc đã làm: 
  - ✅ Hoàn thành viết mã nguồn client bất đồng bộ (`asyncio` & `aiohttp` với `Semaphore=5`).
  - ✅ Hoàn thiện các prompt mẫu tiếng Việt đảm bảo LLM giữ nguyên bản chất pháp lý và phán quyết, chỉ thay đổi thuộc tính nhạy cảm.
  - ✅ Viết thành công bộ regex phát hiện và loại bỏ các phần text thừa/tiêu đề giải thích do LLM tự sinh.
- % hoàn thành: 100% 

---

# Tuần 5
- Từ: 13/4/2026
- Đến: 19/4/2026
- **📋 Báo cáo giữa kỳ: 13/04 - 19/04/2026**
- Công việc cần làm: 
  - Tổng hợp dữ liệu thực nghiệm và kết quả ban đầu của Layer 1.
  - Viết tài liệu báo cáo tiến độ giữa kỳ gửi GVHD.
  - Cập nhật cấu trúc mã nguồn toàn bộ hệ thống (tách biệt các module `layer1`, `layer2`, `validate`, `output`).
- Công việc đã làm: 
  - ✅ Hoàn thành báo cáo giữa kỳ chi tiết về phương pháp sinh dữ liệu đối ngẫu và chỉ số đánh giá CFRE.
  - ✅ GVHD đánh giá cao tiến độ và đồng ý định hướng triển khai sinh dữ liệu Layer 2.
  - ✅ Refactor lại cấu trúc thư mục dự án gọn gàng, modul hóa tốt.
- % hoàn thành: 100% 

---

# Tuần 6
- Từ: 20/4/2026
- Đến: 26/4/2026
- Công việc cần làm: 
  - Chạy pipeline Layer 2 LLM Async Augmentation trên toàn bộ tập dữ liệu.
  - Theo dõi và xử lý các lỗi kết nối API, lỗi timeout trong quá trình chạy.
  - Gộp các kết quả của Job 1 (Rewrite), Job 2 (Attribute Swap), và Job 3 (Classify) với dữ liệu gốc của Layer 1.
  - Xuất file dữ liệu tổng hợp `cf_pairs_all.csv`.
- Công việc đã làm: 
  - ✅ Chạy thành công toàn bộ tiến trình Layer 2 trong khoảng thời gian tối ưu (~7 phút cho gần 8,000 requests) nhờ lập trình bất đồng bộ.
  - ✅ Sửa đổi thành công tên riêng phù hợp giới tính mới (Job 1) và sinh thêm 6,019 cặp cho 5 trục định kiến mới (Job 2).
  - ✅ Gán nhãn thành công cho 721 cặp `unknown` cũ (chủ yếu gán nhãn `legal_fairness`).
  - ✅ Lưu trữ thành công file dữ liệu thô tổng hợp `cf_pairs_all.csv`.
- % hoàn thành: 100% 

---

# Tuần 7
- Từ: 27/4/2026
- Đến: 3/5/2026
- Công việc cần làm: 
  - Xây dựng bộ tiêu chuẩn đánh giá chất lượng dữ liệu gồm 5 tiêu chí chính (`pair_bias_level`, `is_minimal_edit`, `fluency`, `unintended_changes`, `bias_category`).
  - Viết và chạy script LLM Judge đánh giá tự động trên toàn bộ tập dữ liệu thô.
  - Phân tích thống kê tỷ lệ chấp nhận (Acceptance Rates) và phân bố lỗi của các cặp bị loại (Rejected Pairs).
- Công việc đã làm: 
  - ✅ Thiết lập bộ 5 tiêu chí đánh giá khoa học dựa trên các nghiên cứu quốc tế.
  - ✅ Thực thi bộ lọc LLM Judge trên 8,565 cặp dữ liệu, phát hiện có 3,044 cặp bị reject.
  - ✅ Chỉ ra 99.7% lỗi bị reject thuộc về `is_minimal_edit = 0` (lỗi Gender Mismatch R1 và Over-intervention do LLM tự ý thêm bớt từ mô tả stereotype ở trục disability/age).
- % hoàn thành: 100% 

---

# Tuần 8
- Từ: 4/5/2026
- Đến: 10/5/2026
- **🔴 ĐIỂM QUAY & SỰ CỐ MÔ HÌNH SFT**:
  - Thực hiện thử nghiệm huấn luyện SFT (Supervised Fine-Tuning) trên các mô hình ngôn ngữ Qwen (Qwen-0.6B và Qwen-1.7B) nhằm làm tiền đề (pre-requisite) để đưa vào vòng lặp học tự bắt chước RL-SIL.
  - Tuy nhiên, tự phát hiện tác vụ SFT của các mô hình này chưa đúng cấu hình và không phù hợp làm tiền đề cho RL-SIL (lỗi định dạng phân tách prompt-response, dẫn đến mô hình không tạo được trajectory sạch và làm trôi nghĩa pháp lý).
  - Tự nhận định và điều chỉnh hướng nghiên cứu: thay vì tập trung phát triển RL-SIL trên các mô hình sinh thế hệ cũ, quyết định tập trung nghiên cứu sâu vào việc đánh giá thực nghiệm (audit bias, human correlation) và huấn luyện khử định kiến (mitigation) bằng sequence classification.
- Công việc cần làm: 
  - Điều chỉnh kế hoạch nghiên cứu theo định hướng mới.
  - Lên phương án Auto-Fix & Re-filter (Option B) để sửa chữa tự động các cặp dữ liệu bị LLM sinh lỗi.
  - Viết code B1 (Rule-based Regex) sửa lỗi mâu thuẫn danh xưng và tên đệm (Gender Mismatch).
- Công việc đã làm: 
  - ✅ Hoàn thành điều chỉnh đề cương nghiên cứu, tập trung đào sâu phần thực nghiệm kiểm chứng.
  - ✅ Phân tích lỗi SFT của Qwen 0.6B và 1.7B để viết báo cáo giới hạn của mô hình.
  - ✅ Thiết kế luồng xử lý sửa lỗi tự động 3 bước (B1: Regex Title, B2: LLM Rewrite, B3: Re-filter).
  - ✅ Triển khai thành công Regex tự động sửa mâu thuẫn giới tính (ví dụ: đổi "Ông Nguyễn Thị D" ➔ "Bà Nguyễn Thị D").
- % hoàn thành: 100% 

---

# Tuần 9
- Từ: 11/5/2026
- Đến: 17/5/2026
- Công việc cần làm: 
  - Triển khai bước B2 (LLM Rewrite) sử dụng GPT-4o-mini với prompt khắt khe hơn nhằm sửa các lỗi can thiệp quá đà ở trục khuyết tật/tuổi tác.
  - Chạy bước B3 (Re-filter) bằng LLM Judge để kiểm định chất lượng các cặp sau khi sửa đổi.
  - Gộp các dữ liệu accepted và fixed để ra bộ dữ liệu sạch cuối cùng `final_accepted.csv`.
  - Hiệu chỉnh lỗi nạp mô hình Qwen, cấu hình lại prompt template chuẩn hóa và chuyển sang các checkpoint `Qwen2.5-1.5B-Instruct` và `Qwen2.5-3B-Instruct` để làm mô hình nền tảng vững vàng cho vLLM SIL loop.
- Công việc đã làm: 
  - ✅ Sửa chữa thành công các cặp bị over-intervention về dạng chỉnh sửa tối thiểu (minimal edit).
  - ✅ Chạy re-filter thành công, nâng tổng số dữ liệu accepted sạch lên **8,565 cặp đối ngẫu**.
  - ✅ Thống kê phân bố bias cuối cùng: socioeconomic (21.1%), gender (19.4%), age (17.9%), ethnicity (15.1%), religion (13.9%), disability (12.3%), sexual_orientation (0.3%).
  - ✅ Gỡ lỗi thành công cơ chế nạp vLLM, chuẩn hóa xong prompt-response template cho mô hình Qwen mới chọn.
- % hoàn thành: 100% 

---

# Tuần 10 (Thực nghiệm)
- Từ: 18/5/2026
- Đến: 24/5/2026
- Công việc cần làm: 
  - Thiết kế và triển khai quy trình Direction B — Phát Hiện Định Kiến Đa Mô Hình (Multi-Model Bias Audit).
  - Viết script tải và cấu hình 3 mô hình pre-trained: `phobert-base-v2`, `xlm-roberta-base`, `mbert` (và model checkpoint `legal-phobert`).
  - Viết script inference theo batch để trích xuất CLS embedding và log-probabilities.
  - Tính toán điểm số FIS, RCE và CFRE trung bình cho từng mô hình.
- Công việc đã làm: 
  - ✅ Viết thành công mã nguồn đánh giá tĩnh `src/eval/direction_b/`.
  - ✅ Đo đạc thành công chỉ số RCE và FIS trên toàn bộ 8,565 cặp đối ngẫu.
  - ✅ Xuất các báo cáo kết quả chi tiết theo cặp câu và bảng chéo Pivot chéo giữa các mô hình và loại định kiến.
- % hoàn thành: 100% 

---

# Tuần 11 (Thực nghiệm)
- Từ: 25/5/2026
- Đến: 31/5/2026
- Công việc cần làm: 
  - Thiết kế và thực hiện khảo sát đánh giá thủ công của con người (Human Evaluation Study).
  - Viết code lấy mẫu ngẫu nhiên phân tầng (stratified sampling) thu về 100 cặp đối ngẫu đại diện.
  - Gửi biểu mẫu đánh giá và thu thập kết quả từ các annotators con người.
  - Lập trình tính toán hệ số tương quan Spearman $\rho$ và Pearson $r$ giữa điểm số của mô hình và con người.
- Công việc đã làm: 
  - ✅ Viết thành công script sampler thu thập mẫu khảo sát theo đúng tỷ lệ các loại định kiến.
  - ✅ Thu thập và gộp nhãn đánh giá từ các annotators, giải quyết xung đột nhãn thành công.
  - ✅ Kết quả tương quan đạt ý nghĩa thống kê ($p < 0.05$). PhoBERT-base đạt tương quan Spearman cao nhất với con người ($\rho = 0.416$), chứng minh tính thực tiễn của metric CFRE tự tạo.
- % hoàn thành: 100% 

---

# Tuần 12 (Thực nghiệm)
- Từ: 1/6/2026
- Đến: 7/6/2026
- Công việc cần làm: 
  - Thiết kế và triển khai Direction C — Huấn Luyện Khử Định Kiến (Bias Mitigation & Fine-Tuning) phân loại chuỗi nhị phân (nhãn 0 cho `sent_more` chứa bias, nhãn 1 cho `sent_less` phản thực tế) sử dụng PhoBERT.
  - Thiết kế và chạy thực nghiệm Direction F (Proxy Reward) và Direction G (True CFRE Reward) của vòng lặp Self-Imitation Learning (SIL) tích hợp vLLM trên Qwen2.5-1.5B-Instruct và Qwen2.5-3B-Instruct.
  - Thu thập kết quả và so sánh mức độ suy giảm định kiến xã hội (qua chỉ số CFRE, WEAT, SMART) qua các iteration ($\pi_0 \rightarrow \pi_3$).
- Công việc đã làm: 
  - ✅ Hoàn thiện mã nguồn huấn luyện `src/eval/direction_c/` và fine-tune thành công PhoBERT-base trong 3 epochs.
  - ✅ Chạy thành công thực nghiệm SIL với vLLM candidate generator sinh 8 candidates song song cho cả hai model Qwen2.5-1.5B và Qwen2.5-3B.
  - ✅ Kết quả ghi nhận Qwen2.5-3B giảm mạnh điểm định kiến (WEAT và SMART cải thiện rõ rệt), trong khi Qwen2.5-1.5B bị sụp đổ cấu trúc nhắc lệnh (prompt collapse) do dung lượng tham số nhỏ. 
  - ✅ Hoàn thành so sánh đối chiếu giữa Proxy Reward (Direction F) và True CFRE Reward (Direction G), xác định sự mâu thuẫn/hạn chế của metric CFRE khi bị nhiễu bởi copy prompt.
- % hoàn thành: 100% 

---

# Tuần 13
- Từ: 8/6/2026
- Đến: 14/6/2026
- **⏳ Kiểm tra đạo văn: 13/06 - 18/06/2026**
- Công việc cần làm: 
  - Khởi động quá trình kiểm tra đạo văn bài làm trên hệ thống trường.
  - Viết và hoàn thiện các chương lý thuyết của báo cáo nghiên cứu (Chương 1: Giới thiệu & Mục tiêu, Chương 2: Cơ sở lý thuyết & Liên quan, Chương 3: Phương pháp đề xuất).
  - Thiết kế các sơ đồ kiến trúc hệ thống, workflow diagram cho Chương 3.
- Công việc đã làm: 
  - ✅ Nộp bài kiểm tra đạo văn đợt 1 thành công lên hệ thống phòng Sau đại học.
  - ✅ Hoàn thành bản nháp Chương 1, 2 và 3 đạt chuẩn khoa học.
  - ✅ Vẽ xong sơ đồ chi tiết các pipeline Layer 1/2, luồng lọc chất lượng và luồng re-filter Auto-Fix.
- % hoàn thành: 100% 

---

# Tuần 14
- Từ: 15/6/2026
- Đến: 21/6/2026
- **⏳ Kiểm tra đạo văn: 13/06 - 18/06/2026** (hoàn thành)
- **📝 Nộp báo cáo: 16/06 - 23/06/2026** (đang diễn ra)
- **👥 Phân công hội đồng: 16/06 - 21/06/2026**
- Công việc cần làm: 
  - Hoàn thiện Chương 4 (Thực nghiệm & Kết quả) và Chương 5 (Kết luận, Hạn chế & Hướng phát triển).
  - Biên dịch toàn bộ báo cáo nghiên cứu, rà soát lại công thức toán học, lỗi chính tả.
  - Gặp GVHD để ký xác nhận đồng ý cho nộp báo cáo chính thức.
  - Nộp báo cáo chính thức lên cổng thông tin Sau đại học của trường.
- Công việc đã làm: 
  - ✅ Hoàn thành kết quả kiểm tra trùng lặp (đạo văn) đạt yêu cầu (<20%).
  - ✅ Hoàn thiện toàn bộ 5 chương của báo cáo chuyên đề nghiên cứu.
  - ✅ Được GVHD phê duyệt, ký xác nhận báo cáo và nộp thành công file báo cáo PDF lên hệ thống.
- % hoàn thành: 100% 

---

# Tuần 15
- Từ: 22/6/2026
- Đến: 28/6/2026
- **📊 Báo cáo hội đồng: 23/06 - 30/06/2026** (đang diễn ra)
- Công việc cần làm: 
  - Thiết kế slide thuyết trình cuối cùng trước hội đồng phản biện.
  - Xây dựng chương trình Demo chạy thử nghiệm hệ thống (bias audit & mitigation) để trình chiếu trước hội đồng.
  - Luyện tập thuyết trình kỹ lưỡng và chuẩn bị các phương án trả lời câu hỏi phản biện.
- Công việc đã làm: 
  - ✅ Thiết kế xong slide thuyết trình khoa học, chuyên nghiệp (~20 slides).
  - ✅ Hoàn thiện video clip demo chạy pipeline dữ liệu và hiển thị kết quả chênh lệch CFRE trước/sau huấn luyện.
  - ✅ Luyện tập báo cáo thử trong giới hạn thời gian quy định (15-20 phút).
- % hoàn thành: 100% 

---

# Tuần 16
- Từ: 29/6/2026
- Đến: 5/7/2026
- **📊 Báo cáo hội đồng: 23/06 - 30/06/2026** (hoàn thành)
- **✅ Nộp điểm: 01/07 - 03/07/2026**
- Công việc cần làm: 
  - Tiến hành báo cáo thuyết trình chính thức trước Hội đồng đánh giá chuyên đề nghiên cứu.
  - Tiếp thu các ý kiến đóng góp, phản biện của các thầy cô trong hội đồng để chỉnh sửa báo cáo (nếu có).
  - Hoàn thiện các thủ tục hành chính cuối cùng để GVHD nộp điểm chuyên đề về phòng Sau đại học.
- Công việc đã làm: 
  - ✅ Hoàn thành báo cáo chuyên đề xuất sắc trước Hội đồng đánh giá và nhận được phản hồi rất tốt.
  - ✅ Trả lời trôi chảy các câu hỏi Q&A từ các thành viên hội đồng về tính thực tiễn của metric CFRE và phương pháp Auto-Fix Option B.
  - ✅ Hoàn tất việc nộp điểm chuyên đề nghiên cứu (1/7 - 3/7).
- % hoàn thành: 100% 
