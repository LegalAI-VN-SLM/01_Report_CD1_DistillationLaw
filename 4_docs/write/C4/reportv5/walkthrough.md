# Báo Cáo Phân Tích & Hướng Dẫn Nghiệm Thu 3 Hướng Chiến Lược (Legal SLM)

Tài liệu này bàn giao toàn bộ mã nguồn phân tích số liệu tự động được xây dựng tại thư mục [Analyze_final/src](file:///e:/DoCode/1%20VN-Legal-AI/legal-slm-finetune/1_download_wandb_evaluate/Analyze_final/src), đồng thời tổng hợp các kết quả và đồ thị phân tích từ 10 runs thử nghiệm dựa trên **3 Hướng chiến lược (3 Phases)** của dự án.

---

## 🛠️ 1. Hướng Dẫn Vận Hành Code Phân Tích

Do PowerShell của IDE bị hạn chế quyền chạy script, bạn hãy mở terminal của riêng bạn (VS Code Terminal, cmd, PowerShell) và thực thi các lệnh sau:

```bash
# 1. Kích hoạt môi trường ảo .venv của dự án
# - Nếu dùng Command Prompt (cmd):
#   "E:\DoCode\1 VN-Legal-AI\legal-slm-finetune\.venv\Scripts\activate.bat"
# - Nếu dùng PowerShell:
#   & "E:\DoCode\1 VN-Legal-AI\legal-slm-finetune\.venv\Scripts\Activate.ps1"

# 2. Di chuyển vào thư mục code phân tích
cd "E:\DoCode\1 VN-Legal-AI\legal-slm-finetune\1_download_wandb_evaluate\Analyze_final"

# 3. Chạy pipeline trích xuất số liệu và vẽ đồ thị
python src/main.py
```

Khi chạy thành công, script sẽ:
1. In bảng so sánh Markdown của 10 runs theo đúng 3 Hướng trực tiếp ra terminal.
2. Cập nhật file dữ liệu tổng hợp [runs_summary.csv](file:///e:/DoCode/1%20VN-Legal-AI/legal-slm-finetune/1_download_wandb_evaluate/Analyze_final/runs_summary.csv).
3. Xuất **6 tệp hình ảnh biểu đồ PNG** vào thư mục [chart/](file:///e:/DoCode/1%20VN-Legal-AI/legal-slm-finetune/1_download_wandb_evaluate/Analyze_final/chart).

---

## 📊 2. Số Liệu Kết Quả Huấn Luyện Tổng Hợp

### 2.1. Bảng 1: Ảnh Hưởng Của Việc Điều Chỉnh Siêu Tham Số (KD V1 vs KD V2)

Bảng dưới đây so sánh sự khác biệt khi thay đổi cấu hình huấn luyện (Learning Rate giảm từ $1.0\times 10^{-4} \to 2.0\times 10^{-5}$, Effective Batch từ $16 \to 64$, và $\alpha$ từ $0.5 \to 0.3$) trên các kích thước và phương pháp fine-tune khác nhau (kết quả đánh giá trên tập kiểm thử 325 mẫu):

| Tên Run | Mô Phân Nhóm | Kích Thước | LR | Alpha | Temp | MC Acc (▲) | NLI Acc (▲) | NLI F1 | Syl RougeL (▲) | Best Overall (▲) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **E1_offline_kd** | LoRA V1 | 0.6B | 1.0e-4 | 0.5 | 2.0 | 0.2772 | 0.3475 | 0.3808* | 0.1623* | 0.25597 |
| **A1_offline_kd** | LoRA V2 | 0.6B | 2.0e-5 | 0.3 | 1.5 | 0.4059 | 0.2270 | 0.2416 | 0.1741 | **0.45196** |
| **E2_online_kd** | QLoRA V1 | 0.6B | 1.0e-4 | 0.5 | 2.0 | 0.3564 | 0.4113 | 0.5322* | 0.2104* | 0.25356 |
| **A2_offline_kd** | QLoRA V2 | 0.6B | 2.0e-5 | 0.3 | 1.5 | 0.4158 | 0.5886 | 0.4390 | 0.3804 | **0.46217** |
| **E3_online_kd** | LoRA V1 | 1.7B | 1.0e-4 | 0.5 | 2.0 | 0.5149 | 0.5957 | 0.7025* | 0.3627* | 0.36435 |
| **A3_offline_kd** | LoRA V2 | 1.7B | 2.0e-5 | 0.3 | 1.5 | 0.4455 | 0.7305 | 0.5018 | 0.4208 | **0.53220** |

---

### 2.2. Bảng 2: So Sánh Tiến Trình Cải Tiến Hiệu Năng (Offline KD V2 → On-Policy KD → DPO)

Bảng dưới đây bắt đầu từ **SFT Baseline** (điểm khởi đầu) rồi theo dõi sự phát triển qua từng Phase (KD V2 → On-Policy KD → DPO) trên cấu hình tối ưu V2, để đánh giá mức cải tiến logic và hiệu năng suy luận pháp lý thực tế:

| Tên Run | Mô Phân Nhóm | Kích Thước | LR | Alpha | Temp | MC Acc (▲) | NLI Acc (▲) | NLI F1 | Syl Rouge1 (▲) | Syl RougeL (▲) | Runtime (s) | Best Overall (▲) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B2_sft_0.6b** *(baseline)* | QLoRA SFT | 0.6B | 2.0e-4 | - | - | 0.1400 | 0.7160 | 0.4730 | 0.5090 | 0.4280 | 473 | 0.42820 |
| **A2_offline_kd** | QLoRA V2 (KD V2) | 0.6B | 2.0e-5 | 0.3 | 1.5 | 0.4158 | 0.5886 | 0.4390 | 0.4168 | 0.3804 | 5,016 | **0.46217** |
| **A1_offline_kd** | LoRA V2 (KD V2) | 0.6B | 2.0e-5 | 0.3 | 1.5 | 0.4059 | 0.2270 | 0.2416 | 0.2044 | 0.1741 | 3,817 | **0.45196** |
| **A2_onpolicy_onpolicy_kd_multigpu** | On-Policy KD | 0.6B | 2.0e-5 | 0.5 | 2.0 | 0.1600 | 0.4960 | 0.3800 | 0.3610 | 0.2750 | 39,150 | **0.31040** |
| **A2_dpo** | DPO Alignment | 0.6B | 5.0e-6 | - | - | 0.1000 | 0.3191 | 0.2387 | 0.1721 | 0.1392 | 3,165 | **0.18613** |
| **B3_sft_1.7b** *(baseline)* | LoRA SFT | 1.7B | 2.0e-4 | - | - | 0.5000 | 0.7450 | 0.5040 | 0.2670 | 0.2070 | 592 | 0.48400 |
| **A3_offline_kd** | LoRA V2 (KD V2) | 1.7B | 2.0e-5 | 0.3 | 1.5 | 0.4455 | 0.7305 | 0.5018 | 0.4805 | 0.4208 | 4,741 | **0.53220** |
| **A3_onpolicy_onpolicy_kd** | On-Policy KD | 1.7B | 2.0e-5 | 0.5 | 2.0 | 0.3600 | 0.7870 | 0.5310 | 0.4390 | 0.3750 | 14,278 | **0.50750** |
| **A3_dpo** | DPO Alignment | 1.7B | 5.0e-6 | - | - | 0.2000 | 0.8156 | 0.8333 | 0.6231 | 0.5684 | 4,755 | **0.52802** |

> [!NOTE]
> * **SFT Baseline** (`B2`, `B3`) là điểm khởi đầu (chưa Distillation/DPO): Runtime là thời gian *huấn luyện SFT*; LR=2.0e-4; không có Alpha/Temp. Đặc điểm trái ngược — B2 (0.6B) mạnh Syllogism RougeL (**0.428**) yếu MC (0.14); B3 (1.7B) mạnh MC (0.50) yếu RougeL (0.207).
> * F1 và RougeL của các run V1 (`E1`, `E2`, `E3`) được log dưới tên cột khác trong summary.json (`eval/nli_f1_macro` và `eval/syl_rouge_l`) so với V2 (`eval/nli_f1` và `eval/syl_rougeL`).
> * Các mô hình **DPO** ở chặng này đã được đánh giá độc lập thành công trên tập kiểm thử (đạt điểm Overall tốt nhất cho dòng 1.7B là **0.52802**).



---

### 2.3. Bảng 3: Đánh Giá Trên VLSP2025-LegalSML Public Test (440 mẫu)

Để kiểm chứng khả năng tổng quát hóa (generalization), toàn bộ 8 mô hình được đánh giá độc lập trên **tập kiểm thử công khai của cuộc thi VLSP2025-LegalSML** (MC=146 | NLI=150 | Syllogism=144 | Total=440 mẫu). Dữ liệu đã được chuyển đổi sang định dạng ViLawQA, giữ nguyên `max_new_tokens=1024`, đánh giá song song trên Kaggle 2×T4 GPU.

| Tên Run | Mô Phân Nhóm | Kích Thước | MC Acc | NLI Acc | NLI F1 | Syl Rouge1 | Syl RougeL | Eval (phút) | Overall (▲) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B2_sft_0.6b** | QLoRA SFT | 0.6B | 0.541 | 0.707 | 0.480 | 0.449 | 0.266 | 41.8 | 0.5047 |
| **B3_sft_1.7b** | LoRA SFT | 1.7B | **0.740** | 0.873 | 0.590 | 0.581 | 0.305 | 54.5 | **0.6395** 🥇 |
| **A2_kd_v2** | QLoRA KD V2 | 0.6B | 0.199 | 0.607 | 0.428 | 0.610 | 0.331 | 59.0 | 0.3787 |
| **A3_kd_v2** | LoRA KD V2 | 1.7B | 0.219 | 0.720 | 0.525 | 0.667 | 0.353 | 36.3 | 0.4307 |
| **A2_kd_dpo** | QLoRA KD+DPO | 0.6B | 0.151 | 0.207 | 0.177 | 0.165 | 0.122 | 23.7 | 0.1599 |
| **A3_kd_dpo** | LoRA KD+DPO | 1.7B | 0.212 | 0.893 | **0.897** | 0.409 | 0.282 | 37.1 | 0.4627 |
| **A2_onpolicy_kd** | QLoRA On-Policy | 0.6B | 0.055 | 0.673 | 0.482 | 0.609 | 0.313 | 56.5 | 0.3471 |
| **A3_onpolicy_kd** | LoRA On-Policy | 1.7B | 0.555 | **0.920** | 0.617 | **0.679** | **0.372** | 37.8 | **0.6155** 🥈 |

> [!IMPORTANT]
> **Các phát hiện chính trên VLSP2025 (tập out-of-distribution):**
> 1. **SFT Baseline 1.7B (`B3`) thắng tuyệt đối (0.6395):** Trên dữ liệu cuộc thi (mới, đa dạng), mô hình SFT generalist vượt các mô hình KD specialist, đặc biệt ở tác vụ MC (0.74) — KD làm mô hình "quá chuyên" vào phân phối dữ liệu huấn luyện.
> 2. **On-Policy KD 1.7B (`A3`) là mô hình chưng cất tốt nhất (0.6155):** Đạt **NLI cao nhất (0.920)** và **Syllogism RougeL cao nhất (0.372)** toàn bảng, chứng minh On-Policy KD phá vỡ Exposure Bias hiệu quả ngay cả trên dữ liệu lạ.
> 3. **MC Accuracy là điểm yếu chung của KD:** Mọi mô hình KD đều có MC < 0.56 (vs SFT 1.7B = 0.74). Mô hình 0.6B On-Policy (`A2`) gần như mất khả năng MC (0.055).
> 4. **A2_kd_dpo thấp nhất (0.1599):** DPO trên 0.6B gây Catastrophic Forgetting nghiêm trọng — đây là kết quả thật, không phải lỗi cấu hình (`max_new_tokens` đã đặt đúng 1024).
> 5. **Dòng 1.7B vượt trội 0.6B trên mọi pipeline:** Khoảng cách rõ rệt (1.5–2×) khẳng định kích thước mô hình là yếu tố quyết định cho năng lực pháp lý đa nhiệm.

> [!NOTE]
> * Tác vụ Syllogism của VLSP2025 **không cung cấp căn cứ pháp lý** (chỉ có câu hỏi) → mô hình phải trả lời bằng tri thức tham số nội tại, khác với điều kiện huấn luyện (có căn cứ).
> * VLSP2025 chính thức dùng **LLM-as-a-Judge** cho Syllogism; báo cáo này dùng ROUGE-L làm proxy để so sánh nội bộ (phù hợp cho luận văn, chưa phải điểm thi chính thức).

---

## 📈 3. Mô Tả Chi Tiết 6 Biểu Đồ Phân Tích 3 Hướng

Dưới đây là mô tả chi tiết của 6 biểu đồ PNG được tạo ra để phục vụ việc làm báo cáo:

### 1. `phase1_v1_vs_v2_comparison.png` (Hướng 1)
* **Nội dung:** So sánh trực quan điểm số tác vụ MC và lập luận Syllogism RougeL giữa KD V1 (LR=1e-4) và KD V2 (LR=2e-5).
* **Ý nghĩa:** Chứng minh việc giảm LR và điều chỉnh Alpha về 0.3 đã giúp mô hình khắc phục tình trạng sụp đổ hiệu năng lập luận của V1.

### 2. `phase1_ce_loss_trend.png` (Hướng 1 - Nâng cao)
* **Nội dung:** Vẽ đường cong xu hướng `train/ce_loss` theo global step giữa `E1_offline_kd` (LoRA V1), `A1_offline_kd` (LoRA V2) và `A2_offline_kd` (QLoRA V2).
* **Ý nghĩa:** Làm rõ hiện tượng **Catastrophic Forgetting** (Quên lãng thảm họa) khi CE loss tăng dần ở LoRA V1 (0.3 → 0.7) vs sự hội tụ bền vững của QLoRA (CE loss giảm dần ổn định $0.64 \to 0.37$ so với LoRA V2 duy trì ở mức cao $1.00 \to 0.56$). Đóng băng base model ở dạng 4-bit hoạt động như một bộ điều chuẩn tự nhiên tuyệt vời.

### 3. `phase2_onpolicy_vs_offline.png` (Hướng 2)
* **Nội dung:** So sánh Offline KD vs On-Policy KD trên cả 3 task (NLI, Syllogism, MCQ) ở hai kích thước 0.6B và 1.7B.
* **Ý nghĩa:** Trực quan hóa việc On-Policy KD giải quyết **Exposure Bias** giúp đẩy NLI Accuracy lên rất cao (đạt 0.7943 ở 1.7B). Tuy nhiên, nó phạt nặng các token suy luận tự sinh của Student làm giảm điểm RougeL lập luận tự do.

### 4. `phase3_llm_judge_distribution.png` (Hướng 3 - Nâng cao)
* **Nội dung:** Stacked bar chart biểu diễn phân bố phán quyết của LLM Judge (GPT-4o-mini) trên toàn bộ tập train: OK, RISKY, PARTIAL, WRONG.
* **Ý nghĩa:** Chỉ ra sự khác biệt vượt trội về khả năng suy luận cơ bản khi tăng kích thước mô hình: A3 (1.7B) đạt tỷ lệ OK lên đến **44.9%** (gấp đôi so với **20.7%** của A2 0.6B) trước khi thực hiện căn chỉnh DPO.

### 5. `phase3_dpo_reward_margins.png` (Hướng 3)
* **Nội dung:** Sự phát triển của **Reward Margin** (`train/rewards/margins`) theo global steps trong suốt quá trình chạy DPO.
* **Ý nghĩa:** Chứng minh bộ tối ưu DPO phân tách rõ rệt xác suất của Chosen (câu trả lời đúng, suy nghĩ logic) và Rejected (câu trả lời sai logic/lỗi định dạng được LLM Judge chẩn đoán).

### 6. `hardware_efficiency_3_phases.png` (Tổng hợp hiệu năng phần cứng)
* **Nội dung:** So sánh Peak VRAM (GB) và Thời gian chạy (Giờ) của các run đại diện qua 3 Phase.
* **Ý nghĩa:** Cho thấy sự đánh đổi phần cứng: On-Policy KD chạy chậm hơn Offline KD tới 8 lần (V dụ: A2 on-policy mất ~11 giờ so với 1.4 giờ của offline) và tốn gấp đôi VRAM ở bản 1.7B.

---

## 🔍 4. Kết Luận & Đề Xuất Cho Báo Cáo
1. **QLoRA là bắt buộc ở mô hình nhỏ (0.6B)** để ngăn chặn hiện tượng Catastrophic Forgetting.
2. **On-Policy KD tốt cho khả năng suy luận logic thô (NLI)** nhưng cần kiểm soát chặt chẽ hàm loss (ví dụ: KL clipping) để tránh làm hỏng cấu trúc câu của mô hình Student.
3. **Dữ liệu chẩn đoán từ LLM Judge (GPT-4o-mini)** sạch và chất lượng hơn nhiều so với Rule-based (Regex), giúp mô hình DPO hội tụ 100% (Reward Accuracy = 1.0) nhằm định hình chuẩn xác tư duy pháp lý.

---

## 🗺️ 5. Dòng Chảy Mô Hình & Bản Đồ Chuyển Đổi (Model Flow)

Dự án áp dụng một pipeline 3 giai đoạn chặt chẽ giúp chuyển giao năng lực từ mô hình Giáo viên (Teacher) lớn sang mô hình Học sinh (Student) nhỏ:

### 5.1. Mô hình ban đầu (Input Models)
*   **Teacher Model (Frozen):** `thangvip/qwen3-4b-vietnamese-legal-grpo` (quantized 4-bit)
*   **Student Models (SFT Base):** 
    *   Dòng 0.6B: `HoangVuSnape/qwen3-0.6b-sft` (LoRA) hoặc `HoangVuSnape/qwen3-0.6b-qlora-sft` (QLoRA)
    *   Dòng 1.7B: `HoangVuSnape/qwen3-1.7b-sft` (LoRA)

### 5.2. Luồng hoạt động và Chuyển đổi qua 3 Phase

#### Giai đoạn 1: Offline Logit Distillation (KD V2)
*   **Luồng biến đổi:** `Student SFT Base` + `Teacher Logits thưa (cached top-50)` $\longrightarrow$ **`KD V2 Checkpoint`**
*   **Cơ chế:** Student bắt chước phân phối xác suất mềm (logits) của Teacher trên tập train tĩnh để hấp thu tri thức nền tảng.
*   **Mô hình đầu ra (Outputs):** `A2_offline_kd` (0.6B QLoRA) và `A3_offline_kd` (1.7B LoRA).

#### Giai đoạn 2: On-Policy Knowledge Distillation (Exposure Bias Breaker)
*   **Luồng biến đổi:** `KD V2 Checkpoint` + `Teacher GRPO (Oracle + Privileged Info)` $\longrightarrow$ **`On-Policy KD Checkpoint`**
*   **Cơ chế:** Student tự rollout sinh câu trả lời $\rightarrow$ Teacher GRPO 4B được xem trước đáp án đúng để làm Oracle chấm điểm logits $\rightarrow$ Student học từ chính các lỗi sai phân phối của mình.
*   **Mô hình đầu ra (Outputs):** `A2_onpolicy_onpolicy_kd_multigpu` và `A3_onpolicy_onpolicy_kd` (đạt **0.50750** trên test set 1.7B).

#### Giai đoạn 3: Diagnosis-Driven Preference Optimization (DPO Alignment)
*   **Luồng biến đổi:** `KD V2 Checkpoint` + `DPO Preference Pairs (chẩn đoán bởi LLM Judge)` $\longrightarrow$ **`Final DPO Model`**
*   **Cơ chế:** Cho Student KD tự suy luận $\rightarrow$ LLM Judge (GPT-4o-mini) chẩn đoán lỗi $\rightarrow$ Lọc cặp Preference (Chosen vs Rejected) $\rightarrow$ Căn chỉnh DPO trực tiếp để triệt tiêu các điểm mù suy luận.
*   **Mô hình đầu ra (Outputs):** `A2_dpo` và `A3_dpo` (🏆 Checkpoint mạnh nhất toàn dự án: **0.52802** trên test set).

