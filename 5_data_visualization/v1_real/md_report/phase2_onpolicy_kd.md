# Phase 2: On-Policy Knowledge Distillation (Exposure Bias Mitigation)

Phase 2 giải quyết một điểm yếu cốt lõi của phương pháp Offline KD truyền thống bằng cách chuyển dịch sang **On-Policy Knowledge Distillation** (hay còn gọi là Generalized Knowledge Distillation - GKD).

## 1. Vấn đề Exposure Bias trong Lập luận Pháp luật
Trong **Offline KD (Phase 1)**, Student chỉ được huấn luyện trên các chuỗi văn bản tĩnh (gold sequences) do Teacher hoặc con người viết sẵn. Trong quá trình suy luận (Inference), Student lại phải tự sinh ra từng token dựa trên lịch sử dự đoán của chính nó. 

Hiện tượng này gọi là **Exposure Bias (Lệch pha tiếp xúc)**: Student chưa bao giờ được tiếp xúc với những trạng thái "sai lầm" của chính mình trong quá trình train, dẫn đến việc khi sinh từ tự do ở inference, một lỗi nhỏ ban đầu sẽ bị lan truyền lũy kế (error propagation), khiến mô hình suy luận lạc đề, lặp từ hoặc ảo tưởng. Điều này đặc biệt nghiêm trọng trong các bài toán đòi hỏi chuỗi lập luận dài (Reasoning/Thinking block) như miền Luật.

---

## 2. Cơ chế On-Policy KD & Oracle Privileged Information
Để phá vỡ Exposure Bias, On-Policy KD buộc Student phải tự sinh ra câu trả lời trong quá trình train:
1. **Student Rollouts:** Với mỗi prompt $x$, Student tự sinh ra một trajectory lập luận $y^S$ bằng cách lấy mẫu ngẫu nhiên (Temperature sampling $T=0.7$, $p=0.9$).
2. **Oracle Teacher Evaluation:** Teacher (`Qwen3-4B-legal-GRPO`) sẽ nhận prompt $x$ kèm theo **Privileged Information (Thông tin đặc quyền)** là nhãn đáp án đúng (Gold Standard Answer). Nhờ có đáp án chuẩn bên trong prompt, Teacher hoạt động như một Oracle thông thái để đánh giá và sinh phân phối xác suất logits mềm chuẩn xác cho chính chuỗi lập luận lỗi $y^S$ của Student.
3. **KL Divergence:** Student được tối ưu hóa để kéo phân phối xác suất của mình gần với logits của Teacher ngay trên những chuỗi do chính Student sinh ra. Điều này dạy Student cách tự phát hiện và phục hồi khỏi những lỗi suy luận nhỏ.

---

## 3. Token-Level KL Clipping & Sự đánh đổi
### Token-level KL Clipping
Trong quá trình huấn luyện on-policy, các token đặc biệt hoặc các cấu trúc phân cách lập luận (như `<think>`, `</think>`, `<answer>`) thường tạo ra giá trị KL Divergence cực kỳ lớn, lấn át hoàn toàn gradient của các token ngữ nghĩa pháp luật. 

Dự án đã áp dụng kỹ thuật **Token-level KL Clipping** với ngưỡng $C = 5.0$:
$$\text{KL}_{\text{clipped}} = \min(\text{KL}(p^T \parallel p^S), C)$$
Điều này giúp ổn định hóa quá trình huấn luyện và bảo vệ cấu trúc cú pháp của mô hình Student không bị sụp đổ.

### Sự đánh đổi thực nghiệm (So sánh Offline vs On-Policy)
Biểu đồ so sánh giữa Offline KD (V2) và On-Policy KD chỉ ra sự đánh đổi rất lớn:
* **Khả năng suy luận thô (NLI Accuracy) tăng vọt:** Điểm NLI Acc của bản 1.7B được đẩy từ **0.7305 (A3 Offline)** lên **0.7943 (A3 On-Policy)** (đạt kỷ lục tốt nhất).
* **MC Accuracy và Lập luận tự do (Syllogism) bị suy giảm:** Điểm MC Accuracy của 1.7B giảm từ 0.4455 xuống 0.3960. Đặc biệt, Syllogism ROUGE-L bị sụt giảm nặng nề từ **0.4208** xuống còn **0.1340**.
* **Nguyên nhân:** Hàm loss on-policy KL phạt nặng các token suy luận tự do của Student khi nó đi lệch khỏi phân phối của Teacher, vô tình triệt tiêu tính chi tiết và đa dạng trong lập luận tự do (làm suy giảm điểm ROUGE-L).
* **Thời gian huấn luyện cực kỳ đắt đỏ:** On-policy KD đòi hỏi forward-backward liên tục của cả Student và Teacher trên GPU trong quá trình train. Run `A2_onpolicy_onpolicy_kd_multigpu` mất **10.88 giờ** (chạy chậm gấp **8 lần** so với 1.39 giờ của Offline KD V2). Run `A3_onpolicy` tốn tới **45.2 GB Peak VRAM** (so với 22.18 GB của Offline).

---

## 📊 Tham chiếu Biểu đồ Phân tích (Folder `chart/`)
* **`fig3_phase2_onpolicy_vs_offline.png`:** Minh họa rõ rệt sự tăng trưởng mạnh mẽ của NLI Accuracy đi kèm với sự suy giảm điểm số Syllogism ROUGE-L ở cả 2 kích thước mô hình.
* **`fig6_hardware_efficiency_3_phases.png`:** Cho thấy mức tiêu thụ Peak VRAM khổng lồ và thời gian chạy kéo dài vượt trội của các runs On-Policy so với Offline.
