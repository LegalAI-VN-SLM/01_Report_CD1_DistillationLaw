# Bản đồ Tài liệu & Thuật ngữ dự án (Glossary & Document Index)

Chào mừng bạn đến với thư mục tài liệu của dự án **Legal SLM Distillation**. Tài liệu này giúp định hướng nhanh các tài liệu thiết kế, thực nghiệm và các bài báo khoa học nghiên cứu liên quan được lưu trữ trong thư mục [docs/](file:///E:/DoCode/1%20VN-Legal-AI/legal-slm-distillation-pdq/distillation_logit_v2/docs).

---

## 1. Mục lục Tài liệu dự án (Document Index)

### 📂 Thư mục gốc `docs/`

*   **[00_model.md](file:///E:/DoCode/1%20VN-Legal-AI/legal-slm-distillation-pdq/distillation_logit_v2/docs/00_model.md) - Danh sách các mô hình (Models List)**
    *   *Nội dung:* Lưu trữ thông tin và định danh (HuggingFace Hub) của các Student Models (`qwen3-0.6b-sft`, `qwen3-1.7b-sft` bản LoRA/QLoRA) và Teacher Models (`qwen3-4b-vietnamese-legal-grpo`, `SeaLLMs-v3-7B-Chat`).
*   **[01_logit_kd.md](file:///E:/DoCode/1%20VN-Legal-AI/legal-slm-distillation-pdq/distillation_logit_v2/docs/01_logit_kd.md) - Thiết kế Offline Logit KD (Design Spec)**
    *   *Nội dung:* Đặc tả kỹ thuật cho quy trình huấn luyện chưng cất logit ngoại tuyến. Bao gồm kiến trúc xuất logits (Phase 1), huấn luyện KD Loss phối hợp (Phase 2 - $Loss = \alpha L_{KD} + (1-\alpha) L_{CE}$) và cơ chế đánh giá theo từng task (Phase 3).
*   **[02_dpo_sampling.md](file:///E:/DoCode/1%20VN-Legal-AI/legal-slm-distillation-pdq/distillation_logit_v2/docs/02_dpo_sampling.md) - Thiết kế Lấy mẫu Phân lớp DPO (DPO Sampling Spec)**
    *   *Nội dung:* Mô tả phương án xây dựng cặp Preference (`chosen`/`rejected`) đồng nhất định dạng suy nghĩ `<think>` và chiến lược lấy mẫu phân lớp (Stratified Sampling) để rút gọn dữ liệu từ 2.582 mẫu xuống 800 mẫu tiêu biểu nhất.
*   **[03_execution_guide.md](file:///E:/DoCode/1%20VN-Legal-AI/legal-slm-distillation-pdq/distillation_logit_v2/docs/03_execution_guide.md) - Hướng dẫn Chạy thực nghiệm (Execution Guide)**
    *   *Nội dung:* Hướng dẫn chuẩn bị môi trường ảo, kích hoạt conda/venv và các lệnh PowerShell chi tiết để chạy tuần tự các thí nghiệm chưng cất logit trực tuyến (E1) và tối ưu hóa VRAM thông qua cache logit tự động (E2, E3).
*   **[glossary_v2.md](file:///E:/DoCode/1%20VN-Legal-AI/legal-slm-distillation-pdq/distillation_logit_v2/docs/glossary_v2.md) - Từ điển Thuật ngữ V2 (3-phase pipeline)**
    *   *Nội dung:* Từ điển giải nghĩa các khái niệm kỹ thuật sâu ở giai đoạn 2 bao gồm: Exposure Bias, On-Policy KD, Privileged Information (PI), KL Clipping, LLM-as-a-Judge, và các trạng thái chẩn đoán DPO (OK, WRONG, RISKY, PARTIAL).
*   **[glossary_model.md](file:///E:/DoCode/1%20VN-Legal-AI/legal-slm-distillation-pdq/distillation_logit_v2/docs/glossary_model.md) - Từ điển Cấu hình Mô hình (Model Glossary)**
    *   *Nội dung:* Từ điển hệ thống hóa các mã thực nghiệm từ V1 đến V2 (như B1-B3, E1-E7, A1-A3, P1-P3, on-policy và DPO models) cùng chi tiết các siêu tham số huấn luyện tương ứng.

### 📂 Thư mục `docs/reportv1/` (Báo cáo Thử nghiệm V1)

*   **[reportv1/01.md](file:///E:/DoCode/1%20VN-Legal-AI/legal-slm-distillation-pdq/distillation_logit_v2/docs/reportv1/01.md) - Phân tích hiệu năng SFT vs KD**
    *   *Nội dung:* So sánh điểm số thực nghiệm đợt huấn luyện V1 trên 3 tác vụ NLI, Trắc nghiệm (MC) và Tam đoạn luận (Syllogism). Ghi nhận hiện tượng điểm MC tăng mạnh nhưng năng lực lập luận (reasoning) bị giảm sút.
*   **[reportv1/02.md](file:///E:/DoCode/1%20VN-Legal-AI/legal-slm-distillation-pdq/distillation_logit_v2/docs/reportv1/02.md) - Chẩn đoán Lỗi Huấn luyện & Đề xuất Giải pháp**
    *   *Nội dung:* Phân tích chuyên sâu về hiện tượng Catastrophic Forgetting (quên lãng thảm họa) và Alignment Tax. Đề xuất các giải pháp kỹ thuật cụ thể: giảm Learning Rate, tăng Effective Batch Size và tái cấu trúc trọng số hàm loss.

### 📂 Thư mục `docs/docs_paper/` (Tóm tắt & Phân tích các Bài báo khoa học liên quan)

*   **[docs_paper/01.md](file:///E:/DoCode/1%20VN-Legal-AI/legal-slm-distillation-pdq/distillation_logit_v2/docs/docs_paper/01.md) - Distilling Robustness into NLI Models with DTA & DMU**
    *   *Nội dung:* Phân tích bài báo về việc chưng cất độ vững chắc cho các mô hình suy luận. Giới thiệu 2 kỹ thuật: **DTA** (Domain-Targeted Augmentation - sinh dữ liệu không nhãn bằng LLM lớn và dán nhãn mềm từ Teacher) và **DMU** (Distilled Minority Upsampling - tự động tìm lỗi sai của học sinh để tăng trọng số huấn luyện).
*   **[docs_paper/02.md](file:///E:/DoCode/1%20VN-Legal-AI/legal-slm-distillation-pdq/distillation_logit_v2/docs/docs_paper/02.md) - LegalDrill: Diagnosis-Driven Synthesis for Legal Reasoning**
    *   *Nội dung:* Phân tích bài báo LegalDrill ứng dụng cho mô hình ngôn ngữ nhỏ ngành luật. Chi tiết quy trình 4 bước: Exploration $\rightarrow$ Diagnosis (Audit Agent tìm lỗi logic) $\rightarrow$ Targeted Generation (sinh cặp preference Chosen/Rejected đối chiếu) $\rightarrow$ Self-Reflective Verification (lọc mẫu khó bằng logit).
*   **[docs_paper/03.md](file:///E:/DoCode/1%20VN-Legal-AI/legal-slm-distillation-pdq/distillation_logit_v2/docs/docs_paper/03.md) - Training with Harnesses: On-Policy Harness Self-Distillation (OPHSD)**
    *   *Nội dung:* Phân tích phương pháp chưng cất tự thân với bộ khung hỗ trợ suy luận (harness). Giúp mô hình nhỏ tích hợp (internalize) các kỹ năng lập luận quy trình (như Nháp-Xác minh, Lập kế hoạch-Giải quyết) trực tiếp vào trọng số của mình.

### 📂 Thư mục `docs/reportv4/` (Báo cáo Thử nghiệm & Thống kê V4)

*   **[reportv4/01.Pending_Experiments.md](file:///E:/DoCode/1%20VN-Legal-AI/legal-slm-distillation-pdq/distillation_logit_v2/docs/reportv4/01.Pending_Experiments.md) - Kế hoạch hoãn các thực nghiệm Pretrained**
    *   *Nội dung:* Ghi nhận trạng thái thực tế của các thư mục runs và hướng dẫn chạy đối chứng nhóm P1-P3 khi có điều kiện.
*   **[reportv4/02.DPO_Alignment.md](file:///E:/DoCode/1%20VN-Legal-AI/legal-slm-distillation-pdq/distillation_logit_v2/docs/reportv4/02.DPO_Alignment.md) - Phân tích chi tiết pha DPO**
    *   *Nội dung:* Đặc tả kỹ thuật quy trình xây dựng cặp preference (chosen/rejected) và cấu hình huấn luyện DPO bằng Unsloth.
*   **[reportv4/03.Evaluation_Formulas.md](file:///E:/DoCode/1%20VN-Legal-AI/legal-slm-distillation-pdq/distillation_logit_v2/docs/reportv4/03.Evaluation_Formulas.md) - Báo cáo quy trình & công thức đánh giá (Evaluation)**
    *   *Nội dung:* Giải thích chi tiết các công thức toán học của Accuracy, F1-macro, ROUGE-L, và Overall Score.
*   **[reportv4/04.Loss_Formulas.md](file:///E:/DoCode/1%20VN-Legal-AI/legal-slm-distillation-pdq/distillation_logit_v2/docs/reportv4/04.Loss_Formulas.md) - Đặc tả các hàm Loss trong huấn luyện (Loss Formulas)**
    *   *Nội dung:* Mô tả toán học chi tiết cho hàm Loss của Offline KD V2, On-Policy KD (KL clipping) và DPO Preference Alignment.
*   **[reportv4/05.Data_Model_Flow.md](file:///E:/DoCode/1%20VN-Legal-AI/legal-slm-distillation-pdq/distillation_logit_v2/docs/reportv4/05.Data_Model_Flow.md) - Đặc tả dòng chảy Dữ liệu & Mô hình (Data & Model Flow)**
    *   *Nội dung:* Phân tích định dạng đầu vào/đầu ra của tập Raw, Cache Logits, DPO pairs và bảng liên kết các runs trên Wandb.
*   **[reportv4/06.Training_Hardware_Experience.md](file:///E:/DoCode/1%20VN-Legal-AI/legal-slm-distillation-pdq/distillation_logit_v2/docs/reportv4/06.Training_Hardware_Experience.md) - Kinh nghiệm phần cứng & Tối ưu hóa VRAM (Hardware Experience)**
    *   *Nội dung:* Bài học kinh nghiệm huấn luyện trên Kaggle 2xT4, cấu hình thuê GPU ngoài qua Docker và các giải pháp chống OOM VRAM.

---

## 2. Giải nghĩa Thuật ngữ chính (Key Glossary)

| Thuật ngữ | Định nghĩa trong dự án | Tài liệu tham chiếu |
| :--- | :--- | :--- |
| **SFT (Supervised Fine-Tuning)** | Tinh chỉnh có giám sát trên tập dữ liệu pháp luật Việt Nam để làm mô hình nền tảng baseline. | [00_model.md](file:///E:/DoCode/1%20VN-Legal-AI/legal-slm-distillation-pdq/distillation_logit_v2/docs/00_model.md) |
| **KD (Knowledge Distillation)** | Chưng cất tri thức từ Teacher Model sang Student Model bằng cách bắt chước phân bổ xác suất (logits). | [01_logit_kd.md](file:///E:/DoCode/1%20VN-Legal-AI/legal-slm-distillation-pdq/distillation_logit_v2/docs/01_logit_kd.md) |
| **DPO (Direct Preference Optimization)** | Huấn luyện căn chỉnh hành vi mô hình bằng cách tối ưu hóa trực tiếp dựa trên cặp dữ liệu ưa thích (chosen) và không ưa thích (rejected). | [02_dpo_sampling.md](file:///E:/DoCode/1%20VN-Legal-AI/legal-slm-distillation-pdq/distillation_logit_v2/docs/02_dpo_sampling.md) |
| **DTA (Domain-Targeted Augmentation)** | Kỹ thuật sinh dữ liệu không nhãn bổ trợ miền mục tiêu bằng LLM ngoài, sau đó dán nhãn mềm (soft-labels) bằng Teacher để tăng độ vững chắc OOD. | [docs_paper/01.md](file:///E:/DoCode/1%20VN-Legal-AI/legal-slm-distillation-pdq/distillation_logit_v2/docs/docs_paper/01.md) |
| **DMU (Distilled Minority Upsampling)** | Thuật toán tăng trọng số huấn luyện đối với các mẫu dữ liệu thiểu số hoặc mẫu dễ bị lừa/sai lệch logic mà Student thường đoán sai. | [docs_paper/01.md](file:///E:/DoCode/1%20VN-Legal-AI/legal-slm-distillation-pdq/distillation_logit_v2/docs/docs_paper/01.md) |
| **LegalDrill** | Framework tổng hợp dữ liệu hướng chẩn đoán để sửa chữa các điểm mù logic của mô hình nhỏ trong miền pháp luật. | [docs_paper/02.md](file:///E:/DoCode/1%20VN-Legal-AI/legal-slm-distillation-pdq/distillation_logit_v2/docs/docs_paper/02.md) |
| **OPHSD** | Phương pháp tự chưng cất trên chính sách qua bộ khung suy luận giúp SLM "nội hóa" quy trình lập luận phức tạp mà không cần dùng tool ngoài ở bước inference. | [docs_paper/03.md](file:///E:/DoCode/1%20VN-Legal-AI/legal-slm-distillation-pdq/distillation_logit_v2/docs/docs_paper/03.md) |
| **Catastrophic Forgetting** | Hiện tượng mô hình bị quên các tri thức/ cấu trúc câu đã học được ở bước SFT khi trải qua quá trình chưng cất logit với tham số huấn luyện chưa tối ưu. | [reportv1/02.md](file:///E:/DoCode/1%20VN-Legal-AI/legal-slm-distillation-pdq/distillation_logit_v2/docs/reportv1/02.md) |
| **Tam đoạn luận (Syllogism)** | Phương pháp suy luận logic 3 bước kinh điển trong lập pháp: Đại tiền đề $\rightarrow$ Tiểu tiền đề $\rightarrow$ Phán quyết cuối cùng. | [01_logit_kd.md](file:///E:/DoCode/1%20VN-Legal-AI/legal-slm-distillation-pdq/distillation_logit_v2/docs/01_logit_kd.md) |
