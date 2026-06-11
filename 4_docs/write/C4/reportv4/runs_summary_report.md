# Báo Cáo Thực Nghiệm Chi Tiết 10 Lượt Huấn Luyện (Detailed Runs Report)

Báo cáo này thu thập, tổng hợp và phân tích toàn bộ dữ liệu cấu hình, kết quả đánh giá (accuracy, F1, ROUGE-L) và nhật ký lỗi từ các tài liệu `.md` của dự án đối với **10 lượt chạy thực tế (10 Runs)** được ghi nhận trong thư mục `runs/`.

---

## 📌 BẢN ĐỒ PHÂN NHÓM 10 RUNS THỰC TẾ

```
[Giai đoạn 1: KD V1] ──► E1_offline_kd (0.6B LoRA), E2_online_kd (0.6B QLoRA), E3_online_kd (1.7B LoRA)
                               │ (Lỗi Catastrophic Forgetting - LR 1e-4 quá cao)
                               ▼
[Giai đoạn 2: KD V2] ──► A1_offline_kd (0.6B LoRA), A2_offline_kd (0.6B QLoRA), A3_offline_kd (1.7B LoRA)
                               │ (Đã khắc phục: LR 2e-5, Batch 64. A2 QLoRA ổn định nhất)
                               ├────────────────────────────────────────┐
                               ▼                                        ▼
[Giai đoạn 3: On-Policy] ──► A2_onpolicy (0.6B QLoRA)        [Giai đoạn 4: DPO] ──► A2_dpo (0.6B QLoRA)
                             A3_onpolicy (1.7B LoRA)                              A3_dpo (1.7B LoRA)
```

---

## 1. GIAI ĐOẠN 1: KD V1 (E1, E2, E3) — PHÁT HIỆN LỖI HỆ THỐNG

### 1.1. Cấu hình kỹ thuật
*   **Mục tiêu:** Chưng cất logit tuần tự tiếp nối từ checkpoint SFT với LR mặc định cao ($1.0 \times 10^{-4}$), tỷ lệ loss $\alpha = 0.5$, effective batch size 16.
*   **Các run thực tế:**
    1.  **`E1_offline_kd`:** Student `Qwen3-0.6B-sft` + LoRA.
    2.  **`E2_online_kd`:** Student `Qwen3-0.6B-qlora-sft` + QLoRA (Teacher chạy song song tính logit).
    3.  **`E3_online_kd`:** Student `Qwen3-1.7B-sft` + LoRA.

### 1.2. Kết quả đánh giá & Phân tích đường cong huấn luyện
*   **Số liệu so sánh E3_online_kd (1.7B) vs SFT gốc:**
    *   **NLI Accuracy:** SFT (**0.6525**) > KD V1 (0.5957)
    *   **MC Accuracy:** SFT (0.3200) < KD V1 (**0.5149**) *(Tăng mạnh)*
    *   **Syllogism ROUGE-L:** SFT (**0.5952**) > KD V1 (0.3627)
*   **Phân tích lỗi (Catastrophic Forgetting):**
    *   Đường loss của `E3_online_kd` cho thấy mô hình hội tụ cực nhanh chỉ trong 50 step đầu tiên do LR quá cao ($10^{-4}$), phá vỡ hoàn toàn cấu trúc trọng số đã SFT trước đó.
    *   Đường `ce_loss` (học từ nhãn cứng) có xu hướng **tăng ngược trở lại** ở cuối pha training (step 250–326), chứng tỏ mô hình đang quên nhãn chuẩn gốc để đuổi theo phân phối của Teacher.
    *   Mô hình học sinh bị mất khả năng lập luận logic (reasoning) và bắt đầu sinh ra các ký tự rác đa ngôn ngữ (tiếng Anh, Trung, Việt lẫn lộn).

---

## 2. GIAI ĐOẠN 2: KD V2 (A1, A2, A3) — CHIẾN LƯỢC KHẮC PHỤC

### 2.1. Cấu hình kỹ thuật
*   **Thay đổi chính:** Giảm LR 5 lần xuống $2.0 \times 10^{-5}$, tăng effective batch size từ 16 lên 64 (giảm nhiễu gradient), tăng trọng số CE lên $\alpha = 0.3$ (ưu tiên 70% nhãn cứng để neo giữ tri thức cũ).
*   **Các run thực tế:**
    4.  **`A1_offline_kd`:** Student `Qwen3-0.6B-sft` + LoRA.
    5.  **`A2_offline_kd`:** Student `Qwen3-0.6B-qlora-sft` + QLoRA.
    6.  **`A3_offline_kd`:** Student `Qwen3-1.7B-sft` + LoRA.

### 2.2. Kết quả so sánh A1 vs A2 (LoRA vs QLoRA 0.6B)

| Chỉ số / Trạng thái | `A1_offline_kd` (LoRA) | `A2_offline_kd` (QLoRA) | Ý nghĩa thực nghiệm |
| :--- | :---: | :---: | :--- |
| **Đường cong ce_loss** | Tăng dần cuối epoch (0.3 $\to$ 0.7) | Giảm dần ổn định (0.5 $\to$ 0.25) | QLoRA chống Catastrophic Forgetting tốt hơn |
| **Hội tụ** | Đạt đỉnh sớm ở step 60 rồi sụt giảm | Đạt đỉnh ở step 120, xu hướng đi lên | QLoRA ổn định trọng số SFT |
| **Syllogism ROUGE-L** | 0.285 | **0.380** (+33%) | Tăng mạnh khả năng suy luận tự do |
| **Overall Score** | 0.452 | **0.462** | **A2 QLoRA là hướng đi đúng cho 0.6B** |

### 2.3. So sánh định tính 9 mẫu kiểm thử (A2 vs A3)
*   **A2 (0.6B QLoRA) đạt 5/9 câu đúng:** KD sửa thành công lỗi đoán sai của SFT ở câu NLI. Tuy nhiên, mô hình hoàn toàn bất lực trước cả 3 câu Tam đoạn luận (Syllogism 0/3) do giới hạn kích thước mô hình.
*   **A3 (1.7B LoRA) đạt 6/9 câu đúng:** KD sửa thành công lỗi của SFT ở câu trắc nghiệm (MC). Đồng thời giải được 1/3 câu Tam đoạn luận, trích xuất chính xác số liệu pháp lý.

---

## 3. GIAI ĐOẠN 3: ON-POLICY KD (A2_onpolicy, A3_onpolicy)

### 3.1. Cấu hình kỹ thuật
*   **Mục tiêu:** Phá vỡ **Exposure Bias** (độ lệch phơi nhiễm) bằng cách cho Student tự rollout câu trả lời trong quá trình train, Teacher GRPO 4B đóng vai trò là Oracle có đáp án đúng để chấm điểm.
*   **Các kỹ thuật áp dụng:** Cắt xén KL Divergence cấp độ token ở ngưỡng $C=5.0$ để tránh XML token thống trị gradient, sử dụng $\alpha=0.5$.
*   **Các run thực tế:**
    7.  **`A2_onpolicy_onpolicy_kd_multigpu`:** Chạy trên Kaggle T4 Multi-GPU. Phân tách Student ở GPU 0, Teacher 4-bit ở GPU 1. Có khối tự phục hồi lỗi OOM.
    8.  **`A3_onpolicy_onpolicy_kd`:** Chạy trên GPU RTX 6000 Ada (Docker).

### 3.2. Số liệu đánh giá thực tế trên Test Set (325 mẫu)
*   **`A2_onpolicy_onpolicy_kd_multigpu` (Overall: 0.3104):**
    *   NLI Accuracy: **0.4960** | MC Accuracy: **0.1600** | Syllogism ROUGE-L: **0.2750**
*   **`A3_onpolicy_onpolicy_kd` (Overall: 0.5075):**
    *   NLI Accuracy: **0.7870** | MC Accuracy: **0.3600** | Syllogism ROUGE-L: **0.3750**
*   **Nhận xét:** Đánh giá trên test set thực tế cho thấy mô hình 1.7B (`A3_onpolicy`) được cải thiện rõ rệt so với offline KD, đạt **0.5075** (NLI Acc 78.7%, ROUGE-L 37.5%), chứng tỏ hiệu quả của việc phá vỡ Exposure Bias. Tuy nhiên đối với mô hình 0.6B (`A2_onpolicy`), do giới hạn dung lượng tham số, việc tự rollout dẫn đến suy giảm logic nghiêm trọng, điểm tụt xuống còn **0.3104**.


---

## 4. GIAI ĐOẠN 4: DPO ALIGNMENT (A2_dpo, A3_dpo)

### 4.1. Cấu hình kỹ thuật
*   **Mục tiêu:** Căn chỉnh hành vi của Student dựa trên dữ liệu lỗi chẩn đoán bởi GPT-4o-mini (LLM Judge), huấn luyện DPO với LR cực thấp ($5.0 \times 10^{-6}$), hệ số phạt $\beta = 0.1$.
*   **Các run thực tế:**
    9.  **`A2_dpo`:** Căn chỉnh mô hình Qwen3-0.6B từ checkpoint `A2 KD V2`.
    10. **`A3_dpo`:** Căn chỉnh mô hình Qwen3-1.7B từ checkpoint `A3 KD V2`.

### 4.2. Số liệu đánh giá độc lập cuối cùng trên Test Set (325 mẫu)

| Run thực tế | NLI Accuracy | NLI F1-Score | MC Accuracy | Syllogism ROUGE-1 | Syllogism ROUGE-L | Overall Score | Trạng thái hiệu năng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`A2_dpo`** | 0.3191 | 0.2387 | 0.1000 | 0.1721 | 0.1392 | **0.1861** | ❌ **Tụt điểm nặng** (Mô hình 0.6B quá nhỏ bị sập logic khi DPO) |
| **`A3_dpo`** | **0.8156** | **0.8333** | **0.2000** | **0.6231** | **0.5684** | **0.5280** | 🏆 **Checkpoint mạnh nhất** (Tăng mạnh NLI và Syllogism) |

*   **Ý nghĩa học thuật:** DPO đã giải quyết tốt điểm mù logic của mô hình 1.7B (`A3_dpo`), giúp điểm số tổng hợp vượt hẳn các baselines KD V2 trước đó. Tuy nhiên đối với mô hình 0.6B (`A2_dpo`), việc căn chỉnh DPO phản tác dụng do dung lượng tham số quá bé không thể dung hòa được các ràng buộc phạt xác suất của thuật toán.
