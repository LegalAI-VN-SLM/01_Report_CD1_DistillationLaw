# Phase 1: Offline Logit Knowledge Distillation V2

Phase 1 của dự án tập trung vào **Offline Logit Knowledge Distillation (KD V2)**, chuyển giao "tri thức mềm" (dark knowledge) từ mô hình Teacher lớn sang mô hình Student nhỏ thông qua phân phối xác suất logits.

## 1. Cơ chế Chưng cất Logit tĩnh (Offline KD)
Trong Offline KD, mô hình Student được huấn luyện trên một tập dữ liệu tĩnh. Mô hình Teacher (`Qwen3-4B-legal-GRPO`) thực hiện forward pass trên các câu trả lời chuẩn (gold answers) để trích xuất và lưu cache lại top-50 logits có xác suất cao nhất. 

Do cả Student (`Qwen3-0.6B` và `Qwen3-1.7B`) và Teacher đều thuộc cùng dòng mô hình **Qwen3**, chúng chia sẻ chung bộ mã hóa ngôn ngữ (Tokenizer) và từ vựng (Vocabulary). Sự tương thích 1:1 này cho phép so sánh trực tiếp phân phối xác suất logits mà không cần qua lớp chiếu tuyến tính (Projection Layer) để căn chỉnh kích thước.

Hàm loss kết hợp giữa nhãn cứng (Cross-Entropy) và nhãn mềm (Kullback-Leibler Divergence):
$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{KD}} + (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}$$

---

## 2. Bài học từ Thất bại V1: Hiện tượng Quên lãng Thảm họa (Catastrophic Forgetting)
Trong các thử nghiệm đầu tiên (**KD V1** bao gồm `E1_offline_kd`, `E2_online_kd`, `E3_online_kd`), kết quả đánh giá khả năng suy luận logic (Reasoning) của Student bị sụt giảm nghiêm trọng so với mô hình chỉ huấn luyện SFT gốc:
* **Qwen3-0.6B**: Điểm NLI Accuracy của SFT đạt **0.567** nhưng KD V1 chỉ đạt **0.348** (sụt giảm $22\%$).
* **Qwen3-1.7B**: Điểm Syllogism ROUGE-L của SFT đạt **0.595** nhưng KD V1 chỉ đạt **0.363** (sụt giảm $23\%$).

**Nguyên nhân cốt lõi:**
1. **Learning Rate quá cao ($10^{-4}$):** Làm phá vỡ hoàn toàn các trọng số tinh chỉnh mịn (SFT weights) được tối ưu hóa trước đó ngay trong 50 steps đầu tiên.
2. **Effective Batch Size quá nhỏ (16):** Gây nhiễu cực lớn trong quá trình gradient update.
3. **Catastrophic Forgetting:** Đường cong `train/ce_loss` (học từ nhãn cứng của tập huấn luyện gốc) có xu hướng **tăng vọt** từ Epoch 2 về cuối (0.3 $\rightarrow$ 0.7). Điều này chứng minh mô hình đang "unlearn" (quên mất cách trả lời đúng trên nhãn chuẩn) để cố gắng đuổi theo phân phối xác suất mềm của Teacher.

---

## 3. Cải tiến V2: Vai trò Điều chuẩn của QLoRA
Để khắc phục hiện tượng trên, cấu hình **KD V2** đã áp dụng các tinh chỉnh chuyên gia:
* **Hạ thấp Learning Rate:** Giảm xuống còn $2 \times 10^{-5}$ kèm theo Cosine Decay và 10% Warmup steps.
* **Tăng Effective Batch Size:** Đẩy lên 64 (Batch size 4, Gradient Accumulation 16) để ổn định gradient.
* **Tăng cường CE Anchor:** Đặt $\alpha = 0.3$ (tăng trọng số nhãn cứng lên 0.7) để giữ vững kiến thức SFT.

Đặc biệt, sự so sánh giữa **LoRA (A1_offline_kd)** và **QLoRA (A2_offline_kd)** trên mô hình Student 0.6B mang lại insight quan trọng nhất:
* **LoRA (A1):** `ce_loss` vẫn có xu hướng tăng nhẹ về cuối epoch, mô hình bị forgetting nhẹ và thỉnh thoảng sinh rác đa ngôn ngữ.
* **QLoRA (A2):** `ce_loss` hội tụ lý tưởng, giảm dần ổn định qua các epoch ($0.64 \rightarrow 0.37$ so với LoRA V2 duy trì cao $1.00 \rightarrow 0.56$). Điểm số lập luận tam đoạn luận (Syllogism ROUGE-L) tăng vọt từ **0.1741 (A1)** lên **0.3804 (A2)**.

**Giải thích khoa học:** QLoRA thực hiện đóng băng hoàn toàn base weights ở định dạng 4-bit NormalFloat (NF4) và chỉ cập nhật các adapter LoRA 16-bit nhỏ. Cơ chế đóng băng này đóng vai trò như một **bộ điều chuẩn tự nhiên (Implicit Regularization)**. Trọng số gốc của mô hình SFT không thể bị ghi đè hay phá vỡ, ép buộc Student phải học tri thức mới từ Teacher thông qua adapter mà không làm mất đi khả năng lập luận gốc.

---

## 📊 Tham chiếu Biểu đồ Phân tích (Folder `chart/`)
* **`fig1_phase1_v1_vs_v2_comparison.png`:** So sánh trực quan điểm số MC Accuracy và Syllogism ROUGE-L để thấy sự cải thiện vượt bậc của cấu hình V2 so với V1.
* **`fig2_phase1_ce_loss_trend.png`:** Trực quan hóa xu hướng CE Loss trơn (rolling smooth) thể hiện rõ rệt hiện tượng Catastrophic Forgetting ở LoRA V1/V2 và sự hội tụ hoàn hảo của QLoRA V2.
