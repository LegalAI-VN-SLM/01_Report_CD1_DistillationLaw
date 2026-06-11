# Từ Điển Cấu Hình Mô Hình (Model Glossary) — Dự Án Legal SLM Distillation

Tài liệu này hệ thống hóa toàn bộ các ký hiệu, mã thực nghiệm và cấu hình chi tiết từ Giai đoạn 1 (V1), Giai đoạn 2 (V2), On-Policy KD, đến DPO Alignment của dự án. 

---

## 1. So Sánh Tham Số Huấn Luyện Vĩ Mô (V1 vs V2)

Để giải quyết các vấn đề về độ nhiễu gradient và hiện tượng **Catastrophic Forgetting** (Quên lãng thảm họa) ở V1, cấu hình huấn luyện đã được điều chỉnh như sau:

| Siêu tham số (Hyperparameter) | Phiên bản V1 (Gốc) | Phiên bản V2 (Cải tiến) | Ý nghĩa / Mục tiêu cải tiến |
| :--- | :---: | :---: | :--- |
| **Learning Rate (LR)** | $1.0 \times 10^{-4}$ | $2.0 \times 10^{-5}$ | Giảm 5 lần để tránh phá vỡ trọng số đã SFT mịn màng của Student. |
| **Loss Weight $\alpha$** | 0.5 (50% KD + 50% CE) | 0.3 (30% KD + 70% CE) | Tăng tỷ lệ Cross-Entropy với nhãn chuẩn để kiềm chế mô hình không quên kiến thức gốc. |
| **Temperature $T$** | 2.0 | 1.5 | Giảm nhiệt độ giúp phân phối xác suất của Teacher tập trung và sắc nét hơn. |
| **Effective Batch Size** | 16 | 64 | Tăng gấp 4 lần để làm mịn đường cong loss, giảm thiểu độ nhiễu gradient. |
| **Max Epochs** | 2 | 3 | Tăng số epoch vì Learning Rate thấp hơn cần nhiều bước huấn luyện hơn để hội tụ. |
| **Early Stopping Patience** | 2 | 3 | Tăng độ kiên nhẫn để tránh dừng sớm khi học với tốc độ chậm. |
| **Eval Steps** | 100 | 60 | Đánh giá dày hơn để nhanh chóng bắt được checkpoint tốt nhất. |

---

## 2. Danh Sách Các Mã Thử Nghiệm Chi Tiết (V1 & V2)

### 2.1. Nhóm SFT Baselines (Đánh giá đối chứng, không huấn luyện KD)
Các checkpoint được lấy trực tiếp từ HuggingFace Hub làm điểm xuất phát (nền tảng) cho các thực nghiệm chưng cất:
*   **`B1`**: Qwen3-0.6B (LoRA SFT) — Chạy trên checkpoint SFT gốc.
*   **`B2`**: Qwen3-0.6B (QLoRA SFT) — Chạy trên checkpoint SFT lượng tử hóa.
*   **`B3`**: Qwen3-1.7B (LoRA SFT) — Chạy trên checkpoint SFT gốc của dòng mô hình lớn hơn.

---

### 2.2. Nhóm KD V1 (Chưng cất từ SFT checkpoint — Cấu hình cũ)
*   **`E1`**: Qwen3-0.6B + LoRA KD. LR = $10^{-4}$, Batch = 8, Grad Accum = 2 (Effective = 16). Hub: `HoangVuSnape/qwen3-0.6b-legal-kd-lora`
*   **`E2`**: Qwen3-0.6B + QLoRA KD. LR = $10^{-4}$, Batch = 8, Grad Accum = 2 (Effective = 16). Hub: `HoangVuSnape/qwen3-0.6b-legal-kd-qlora`
*   **`E3`**: Qwen3-1.7B + LoRA KD. LR = $10^{-4}$, Batch = 4, Grad Accum = 4 (Effective = 16). Hub: `HoangVuSnape/qwen3-1.7b-legal-kd-lora`

### 2.3. Nhóm KD V1 từ Pretrained (Không qua SFT, huấn luyện từ base model)
*   **`E5`**: Qwen3-0.6B (Base) + LoRA KD. Hub: `HoangVuSnape/qwen3-0.6b-legal-kd-from-pretrained-lora`
*   **`E6`**: Qwen3-0.6B (Base) + QLoRA KD. Hub: `HoangVuSnape/qwen3-0.6b-legal-kd-from-pretrained-qlora`
*   **`E7`**: Qwen3-1.7B (Base) + LoRA KD. Hub: `HoangVuSnape/qwen3-1.7b-legal-kd-from-pretrained-lora`

---

### 2.4. Nhóm KD V2 (Chưng cất từ SFT checkpoint — Cấu hình cải tiến)
*   **`A1`**: Qwen3-0.6B + LoRA KD V2. LR = $2 \times 10^{-5}$, Batch = 4, Grad Accum = 16 (Effective = 64). Hub: `HoangVuSnape-CD1/qwen3-0.6b-legal-kd-v2-lora`
*   **`A2`**: Qwen3-0.6B + QLoRA KD V2. LR = $2 \times 10^{-5}$, Batch = 4, Grad Accum = 16 (Effective = 64). Hub: `HoangVuSnape-CD1/qwen3-0.6b-legal-kd-v2-qlora`
*   **`A3`**: Qwen3-1.7B + LoRA KD V2. LR = $2 \times 10^{-5}$, Batch = 2, Grad Accum = 32 (Effective = 64). Hub: `HoangVuSnape-CD1/qwen3-1.7b-legal-kd-v2-lora`

### 2.5. Nhóm KD V2 từ Pretrained (Huấn luyện từ base model — Đang hoãn/Pending)
*   **`P1`**: Qwen3-0.6B (Base) + LoRA KD V2. LR = $5 \times 10^{-5}$, Batch = 4, Grad Accum = 16 (Effective = 64). Hub: `HoangVuSnape-CD1/qwen3-0.6b-legal-kd-pretrained-v2-lora`
*   **`P2`**: Qwen3-0.6B (Base) + QLoRA KD V2. LR = $5 \times 10^{-5}$, Batch = 4, Grad Accum = 16 (Effective = 64). Hub: `HoangVuSnape-CD1/qwen3-0.6b-legal-kd-pretrained-v2-qlora`
*   **`P3`**: Qwen3-1.7B (Base) + LoRA KD V2. LR = $5 \times 10^{-5}$, Batch = 2, Grad Accum = 32 (Effective = 64). Hub: `HoangVuSnape-CD1/qwen3-1.7b-legal-kd-pretrained-v2-lora`

---

## 3. Nhóm Thực Nghiệm On-Policy KD & DPO

### 3.1. Giai đoạn 2: On-Policy KD từ GRPO Teacher
Student tự sinh phản hồi $\rightarrow$ Teacher GRPO (4B) đọc và chấm điểm logits (sử dụng cơ chế Oracle với Privileged Information).
*   **`A2_onpolicy`**: Khởi chạy từ `A2 KD V2` (0.6B). LR = $2.0 \times 10^{-5}$, Batch = 4, Grad Accum = 8 (Effective = 32). $\alpha = 0.5$ (50% KD + 50% CE). KL clipping = 5.0. Hub: `HoangVuSnape-CD1/qwen3-0.6b-legal-onpolicy-kd-qlora`
*   **`A3_onpolicy`**: Khởi chạy từ `A3 KD V2` (1.7B). LR = $2.0 \times 10^{-5}$, Batch = 8, Grad Accum = 4 (Effective = 32). $\alpha = 0.5$ (50% KD + 50% CE). KL clipping = 5.0. Hub: `HoangVuSnape-CD1/qwen3-1.7b-legal-onpolicy-kd-lora`

### 3.2. Giai đoạn 3: DPO Preference Alignment (Căn chỉnh lỗi dựa trên chẩn đoán)
Huấn luyện tối ưu hóa ưa thích DPO dựa trên dữ liệu chẩn đoán sai lệch từ LLM Judge.
*   **`A2_dpo`**: Căn chỉnh trên mô hình 0.6B. Khởi chạy từ `A2 KD V2` (0.6B). Beta = 0.1, LR = $5.0 \times 10^{-6}$, 2 Epochs. Hub: `HoangVuSnape-CD1/qwen3-0.6b-legal-kd-dpo-qlora`
*   **`A3_dpo`**: Căn chỉnh trên mô hình 1.7B. Khởi chạy từ `A3 KD V2` (1.7B). Beta = 0.1, LR = $5.0 \times 10^{-6}$, 2 Epochs. Hub: `HoangVuSnape-CD1/qwen3-1.7b-legal-kd-dpo-lora`

---

## 4. Tóm Tắt Bản Đồ Checkpoint Cuối Cùng Trên HuggingFace

| Tên mã hóa | Checkpoint trên HuggingFace Hub | Vai trò thực nghiệm | Trạng thái chạy |
| :--- | :--- | :--- | :---: |
| **`A1 KD`** | `HoangVuSnape-CD1/qwen3-0.6b-legal-kd-v2-lora` | Offline KD V2 (LoRA 0.6B) | ✅ Hoàn thành |
| **`A2 KD`** | `HoangVuSnape-CD1/qwen3-0.6b-legal-kd-v2-qlora` | Offline KD V2 (QLoRA 0.6B) | ✅ Hoàn thành |
| **`A3 KD`** | `HoangVuSnape-CD1/qwen3-1.7b-legal-kd-v2-lora` | Offline KD V2 (LoRA 1.7B) | ✅ Hoàn thành |
| **`A2 On-Policy`** | `HoangVuSnape-CD1/qwen3-0.6b-legal-onpolicy-kd-qlora` | On-Policy KD (0.6B) | ✅ Hoàn thành |
| **`A3 On-Policy`** | `HoangVuSnape-CD1/qwen3-1.7b-legal-onpolicy-kd-lora` | On-Policy KD (1.7B) | ✅ Hoàn thành |
| **`A2 DPO`** | `HoangVuSnape-CD1/qwen3-0.6b-legal-kd-dpo-qlora` | DPO Alignment (0.6B) | ✅ Hoàn thành |
| **`A3 DPO`** | `HoangVuSnape-CD1/qwen3-1.7b-legal-kd-dpo-lora` | DPO Alignment (1.7B) | ✅ Hoàn thành |
