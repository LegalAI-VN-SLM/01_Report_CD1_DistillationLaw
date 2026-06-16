# Phase 3: Diagnosis-Driven Preference Optimization (DPO Alignment)

Phase 3 là chặng cuối cùng của pipeline, thực hiện căn chỉnh hành vi ưu tiên (Alignment) bằng phương pháp **Direct Preference Optimization (DPO)** dựa trên kết quả chẩn đoán lỗi của LLM Judge.

## 1. Ý tưởng Căn chỉnh Ưu tiên (Preference Alignment)
Mặc dù chưng cất logit giúp Student tiếp thu tri thức xác suất từ Teacher, mô hình Student vẫn gặp nhiều lỗi về định dạng lập luận (như cấu trúc `<think>` bị cắt cụt, lặp từ vô hạn, hoặc ảo tưởng luật pháp). 

DPO căn chỉnh trực tiếp xác suất của Student trên các cặp phản hồi ưa thích:
* **Chosen ($y_w$):** Câu trả lời đúng chuẩn (gồm cả chuỗi suy nghĩ logic).
* **Rejected ($y_l$):** Câu trả lời bị lỗi do chính Student sinh ra.

Mô hình sẽ học cách tăng xác suất sinh câu trả lời chosen và đồng thời giảm xác suất sinh câu trả lời rejected.

> [!IMPORTANT]
> **Lưu ý cốt lõi về nguồn dữ liệu và base model của Phase 3:**
> 1. **Nguồn gốc dữ liệu Preference:** Toàn bộ tập dữ liệu Preference DPO được xây dựng bằng cách cho mô hình **Student 0.6B QLoRA chưng cất Offline (A2_offline_kd)** tự sinh câu trả lời trên tập huấn luyện. Sau đó, LLM Judge (GPT-4o-mini) tiến hành chẩn đoán xem mô hình 0.6B này bị sai ở đâu (WRONG, RISKY) để làm cơ sở xây dựng các cặp preference (chosen = gold answer, rejected = câu trả lời sai của Student 0.6B). Tập dữ liệu này được sử dụng chung để huấn luyện căn chỉnh DPO cho cả 0.6B (`A2_dpo`) và 1.7B (`A3_dpo`).
> 2. **Base Checkpoint huấn luyện DPO:** Cả hai mô hình DPO (`A2_dpo` và `A3_dpo`) đều sử dụng base checkpoint trực tiếp từ chặng **Offline KD V2** (`A2_offline_kd` cho 0.6B và `A3_offline_kd` cho 1.7B) để thực hiện căn chỉnh DPO. Dự án **chưa** sử dụng checkpoints của chặng Online/On-Policy KD làm base model cho DPO ở giai đoạn này.

---

## 2. LLM-as-a-Judge (GPT-4o-mini) vs Rule-Based Diagnosis
Thay vì sử dụng bộ đánh giá cứng nhắc dựa trên luật (Rule-based Regex/ROUGE-L) - vốn rất dễ đánh oan các câu trả lời đúng ngữ nghĩa nhưng sai định dạng (false negative) hoặc bỏ sót các suy luận sai logic thực chất (false positive), dự án đã xây dựng pipeline **LLM Judge** sử dụng `GPT-4o-mini` để đọc hiểu và chẩn đoán toàn diện.

LLM Judge đánh giá cả **Answer** và **Thinking block** dựa trên ngữ nghĩa thực tế, phân loại đầu ra của Student thành 4 nhóm chẩn đoán chính xác:
1. **OK (Đạt toàn diện):** Đáp án đúng và tư duy suy luận logic tốt.
2. **RISKY (Đoán mò):** Đáp án đúng nhưng tư duy suy luận bị sai hoặc rỗng.
3. **PARTIAL (Một phần):** Trả lời đúng một phần hoặc sai nhẹ về số liệu/điều luật.
4. **WRONG (Sai hoàn toàn):** Sai cả đáp án lẫn lập luận logic.

Tập dữ liệu preference DPO được xây dựng cực kỳ sạch bằng cách giữ lại các mẫu thuộc nhóm **WRONG** và **RISKY** để làm chuỗi `rejected`.

---

## 3. Phân tích Kết quả Thực nghiệm
### So sánh Khả năng Suy luận qua con mắt LLM Judge (A2 vs A3)
Sự so sánh tỷ lệ chẩn đoán trên tập huấn luyện mang lại insight quan trọng về kích thước mô hình (Model Size):
* Mô hình **A3 (1.7B LoRA)** đạt tỷ lệ **OK** lên tới **$44.9\%$** (gấp đôi so với **$20.7\%$** của bản **A2 0.6B QLoRA**).
* Tỷ lệ sai hoàn toàn (**WRONG**) giảm mạnh từ **$63.5\%$ (A2)** xuống **$46.0\%$ (A3)**.
* Sự cải tiến này thể hiện rõ rệt ở tác vụ khó nhất là **Syllogism (Tam đoạn luận)**: A3 đạt tỷ lệ OK là **$20.0\%$** (so với chỉ **$6.0\%$** của A2). 
* Do đó, mô hình 1.7B tạo ra tập DPO ít cặp preference hơn nhưng có chất lượng và độ sạch cao hơn hẳn (1,434 pairs của A3 so với 2,065 pairs của A2).

### Sự tăng trưởng của Reward Margin
Kết quả huấn luyện DPO trên Kaggle sử dụng Unsloth + TRL cho thấy sự hội tụ lý tưởng:
* **Reward Accuracy đạt 1.0 (100%):** Trình tối ưu hóa DPO học cách phân tách hoàn toàn Chosen và Rejected.
* **Reward Margin tăng trưởng vững chắc:** Khoảng cách điểm thưởng giữa Chosen và Rejected tăng liên tục theo steps huấn luyện (đạt từ 25.0 đến 33.0 ở cuối chặng). Điều này đảm bảo mô hình đã học được cấu trúc suy nghĩ chuẩn xác và từ bỏ hành vi sinh lập luận lỗi.

---

## 📊 Tham chiếu Biểu đồ Phân tích (Folder `chart/`)
* **`fig4_phase3_llm_judge_distribution.png`:** Trực quan hóa tỷ lệ phần trăm phân bố chẩn đoán 4 nhóm của LLM Judge, làm rõ sự tiến bộ của mô hình 1.7B so với 0.6B.
* **`fig5_phase3_dpo_reward_margins.png`:** Đồ thị đường (Line Chart) biểu diễn sự tăng trưởng mạnh mẽ của Reward Margin theo global steps cho cả hai run `A2_dpo` và `A3_dpo`.
