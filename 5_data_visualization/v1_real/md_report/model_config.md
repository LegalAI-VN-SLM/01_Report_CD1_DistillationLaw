# Model & Hyperparameter Configurations (Cấu hình Mô hình và Siêu tham số)

Tài liệu này tổng hợp toàn bộ thông tin chi tiết về cấu hình mô hình (Teacher & Student), các tham số PEFT (LoRA/QLoRA) và siêu tham số huấn luyện qua cả 3 Phases thực nghiệm để phục vụ viết báo cáo/luận văn.

---

## 1. Thông tin Mô hình gốc (Base Models)

### Mô hình Giáo viên (Teacher Model)
* **Tên mô hình:** `thangvip/qwen3-4b-vietnamese-legal-grpo`
* **Đặc điểm:** Mô hình chuyên biệt về suy luận pháp lý tiếng Việt và cấu trúc lập luận tam đoạn luận (Syllogism), được tối ưu hóa qua phương pháp GRPO (Group Relative Policy Optimization).
* **Trạng thái:** Đóng băng hoàn toàn (Frozen), được load ở định dạng quantized 4-bit (NF4) để chạy forward pass xuất logits nhằm tiết kiệm VRAM.

### Mô hình Học sinh (Student Models)
* **Kích thước 0.6B:** `HoangVuSnape/qwen3-0.6b-sft` (Đã được huấn luyện SFT trên dữ liệu pháp lý).
* **Kích thước 1.7B:** `HoangVuSnape/qwen3-1.7b-sft` (Đã được huấn luyện SFT trên dữ liệu pháp lý).

---

## 2. Cấu hình PEFT (Parameter-Efficient Fine-Tuning)

Để thực hiện huấn luyện trên hạ tầng giới hạn, dự án áp dụng phương pháp LoRA và QLoRA thông qua thư viện **Unsloth**:

| Tham số cấu hình | Thiết lập LoRA (A1, A3) | Thiết lập QLoRA (A2) |
| :--- | :--- | :--- |
| **LoRA Rank ($r$)** | 64 | 64 |
| **LoRA Alpha ($\alpha$)** | 128 | 128 |
| **Target Modules** | Tất cả các lớp Linear chính | Tất cả các lớp Linear chính |
| **LoRA Dropout** | 0.0 (Tối ưu hóa bởi Unsloth) | 0.0 (Tối ưu hóa bởi Unsloth) |
| **Quantization Type** | Không quantize base model (16-bit) | **4-bit NF4 (NormalFloat4)** |
| **Double Quantization** | Không áp dụng | **Enabled** (Tiết kiệm thêm VRAM) |
| **Compute Dtype** | `bfloat16`/`float16` | `bfloat16`/`float16` |

* Các Target Modules chi tiết bao gồm: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

---

## 3. Siêu tham số huấn luyện chi tiết qua 3 Phases

### 📍 Phase 1: Offline Logit Knowledge Distillation V2 (Học Logit tĩnh)
Mục tiêu là chưng cất phân phối xác suất mềm từ Teacher sang Student với cấu hình LR thấp và Batch size lớn để chống Catastrophic Forgetting.
* **Learning Rate (LR):** $2 \times 10^{-5}$
* **Learning Rate Scheduler:** `cosine` với 10% Warmup steps.
* **Effective Batch Size:** 64
  * *Bản 0.6B (A2):* Batch size = 4, Gradient Accumulation = 16.
  * *Bản 1.7B (A3):* Batch size = 2, Gradient Accumulation = 32.
* **Số Epochs:** 3 Epochs.
* **Trọng số hàm Loss ($\alpha$):** $\alpha = 0.3$ (30% KL Divergence nhãn mềm từ Teacher, 70% Cross-Entropy nhãn cứng từ Gold labels).
* **Nhiệt độ chưng cất ($T$):** 1.5
* **Sparse KD Top-K:** 50 (Chỉ tính KL Divergence trên top-50 tokens có xác suất cao nhất của Teacher để tiết kiệm 99% VRAM).

### 📍 Phase 2: On-Policy Knowledge Distillation (Phá vỡ Exposure Bias)
Student tự sinh câu trả lời trong quá trình train và học trực tiếp từ logits của Teacher trên các quỹ đạo tự sinh đó.
* **Learning Rate (LR):** $2 \times 10^{-5}$
* **Effective Batch Size:** 32
  * *Bản 0.6B (A2):* Batch size = 4, Gradient Accumulation = 8.
  * *Bản 1.7B (A3):* Batch size = 8, Gradient Accumulation = 4.
* **Trọng số hàm Loss ($\alpha$):** $\alpha = 0.5$ (Cân bằng 50% KL Divergence và 50% Cross-Entropy nhãn cứng).
* **Nhiệt độ chưng cất ($T$):** 2.0
* **KL Clipping Threshold ($C$):** 5.0 (Cắt cụt KL divergence ở các token cấu trúc `<think>` để ổn định gradient).
* **Decode Generator Config:** Temperature Sampling ($T=0.7$, $top\_p=0.9$).

### 📍 Phase 3: Diagnosis-Driven DPO (Căn chỉnh Ưu tiên)
Giai đoạn DPO được thực hiện để căn chỉnh cấu trúc suy nghĩ và định dạng câu trả lời dựa trên tập dữ liệu preference chẩn đoán lỗi logic từ LLM Judge.

#### 1. DPO được thực hiện từ Model nào?
DPO được huấn luyện trực tiếp tiếp nối từ các **Checkpoint chưng cất tốt nhất (Best Offline KD Checkpoints)** của mô hình học sinh ở Phase 1:
* **Với mô hình A2 (0.6B):** Dùng checkpoint `HoangVuSnape-CD1/qwen3-0.6b-legal-kd-v2-qlora` (A2_offline_kd).
* **Với mô hình A3 (1.7B):** Dùng checkpoint `HoangVuSnape-CD1/qwen3-1.7b-legal-kd-v2-lora` (A3_offline_kd).

> [!NOTE]
> **Cơ chế nạp Model trong `04_train_dpo.py`:**
> Do checkpoint KD trước đó đã chứa sẵn LoRA/QLoRA adapter, thư viện `Unsloth` sẽ nạp trực tiếp mô hình qua hàm `FastLanguageModel.from_pretrained()` ở trạng thái có thể huấn luyện (trainable). Mã nguồn **không cần** gọi lại hàm `get_peft_model()`, tránh việc tạo thêm chồng chéo LoRA adapter.

#### 2. DPO lấy dữ liệu gì để Huấn luyện (DPO Pairs)?
Dữ liệu DPO được sinh ra từ chính các lỗi sai của Student thông qua quy trình 3 bước tự động:
1. **Bước 1: Chạy suy luận của Student (`01_student_inference.py`):**
   Cho mô hình học sinh (KD best checkpoint) chạy suy luận trên toàn bộ tập train gốc (`vilawqa_sft_train.jsonl`). Đầu ra lưu tại thư mục `data/dpo_student/01_inference/`.
2. **Bước 2: Chẩn đoán lỗi của Student (`02_diagnose.py`):**
   So sánh đầu ra tự sinh của Student (`student_output`) với nhãn chuẩn (`ground_truth`). Thuật toán chấm điểm độc lập cả **Đáp án cuối cùng (Answer)** và **Chuỗi suy nghĩ (Thinking)** để phân loại thành 4 nhóm:
   * `OK` (Answer đúng, suy nghĩ tốt): *Không dùng cho DPO.*
   * `RISKY` (Answer đúng nhưng suy nghĩ tệ - đoán mò): **Ứng viên DPO.**
   * `PARTIAL` (Answer sai nhưng suy nghĩ tốt - đi đúng hướng nhưng kết luận sai): **Ứng viên DPO.**
   * `WRONG` (Sai cả answer lẫn suy nghĩ): **Ứng viên DPO.**
   
   *(Trong thực tế, bạn đã thay thế bộ chấm Regex/ROUGE-L mặc định bằng bộ chấm nâng cao **LLM Judge (GPT-4o-mini)** để tăng độ chính xác và bắt được các lỗi ngữ nghĩa).*
3. **Bước 3: Tạo cặp Preference (`03_build_dpo_pairs.py`):**
   Từ danh sách ứng viên lỗi ở trên, mã nguồn tiến hành đóng gói thành cấu trúc cặp Preference chuẩn của TRL:
   * `prompt`: Câu hỏi gốc (`instruction` + `input`).
   * `chosen`: Đáp án chuẩn mực từ tập SFT gốc (chứa suy nghĩ `<think>` chuẩn).
   * `rejected`: Câu trả lời sai thực tế do Student tự sinh ra (`student_output`).
   
   Sau đó, tập dữ liệu này được lọc và lấy mẫu phân lớp (**Stratified Sampling**) cố định Seed 42 để giữ nguyên tỷ lệ task và format, tạo ra tập dữ liệu rút gọn gồm **800 mẫu tiêu biểu nhất** lưu tại thư mục `data/data_dpo_final/`.

#### 3. Cấu hình huấn luyện DPO được thiết lập ra sao?
Cấu hình huấn luyện chi tiết nằm trong `04_train_dpo.py`:
* **Tối ưu hóa phần cứng (Unsloth Patch):**
  Sử dụng `PatchDPOTrainer()` của Unsloth để tăng tốc độ huấn luyện lên gấp **2 lần** và tối ưu hóa bộ nhớ. Đặc biệt, tham số `ref_model=None` được truyền vào `DPOTrainer` giúp Unsloth tự tính toán reference logits ngầm mà không cần nạp thêm một bản sao model thứ hai vào VRAM (giảm 50% tải VRAM).
* **Các siêu tham số huấn luyện chính (DPO Hyperparameters):**
  * `beta = 0.1`: Hệ số phạt DPO (DPO temperature). Kiểm soát độ lệch so với mô hình tham chiếu.
  * `learning_rate = 5.0e-6` (hoặc `5e-6`): Tốc độ học **cực kỳ thấp** để bảo toàn tối đa các trọng số KD đã học ở giai đoạn trước, tránh làm hỏng năng lực lập luận logic chung.
  * `lr_scheduler_type = "cosine"`: Giảm dần lr theo hàm cosine.
  * `warmup_ratio = 0.1`: Khởi động ấm 10% tổng số bước.
  * `max_epochs = 2`: Chạy huấn luyện trong 2 epoch.
  * `max_length = 2048` & `max_prompt_length = 1024`.
  * `optimizer = "adamw_8bit"`: Thuật toán tối ưu hóa 8-bit nhằm tiết kiệm VRAM trên Kaggle T4.
  * **Effective Batch Size:** 16 (Bản 0.6B: Batch size = 2, Grad Accum = 8; Bản 1.7B: Batch size = 1, Grad Accum = 16).
* **Phân chia dữ liệu:**
  Chia ngẫu nhiên theo tỷ lệ **95% Train / 5% Validation** (hoặc 90/10 tùy cấu hình chạy thực tế) để theo dõi `eval_loss` và chọn ra checkpoint tốt nhất tránh hiện tượng overfitting.
