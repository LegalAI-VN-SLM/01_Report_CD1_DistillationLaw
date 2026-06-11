# 📖 Glossary of Models (Thuật ngữ & Cấu hình Mô hình)

Tài liệu này tổng hợp chi tiết thông tin định nghĩa, cấu hình phần cứng, tham số huấn luyện, định dạng đầu vào (Input) và đầu ra (Output) của tất cả các mô hình được sử dụng trong dự án theo báo cáo nghiệm thu [walkthrough.md](file:///e:/DoCode/1%20VN-Legal-AI/legal-slm-finetune/1_download_wandb_evaluate/report/v2_report_full/walkthrough.md).

---

## 1. Teacher Model (Mô hình Giáo viên)

*   **Tên mô hình gốc:** `thangvip/qwen3-4b-vietnamese-legal-grpo`
*   **Kích thước:** 4 Billion Parameters
*   **Mô tả:** Mô hình cơ sở (Base model) chuyên biệt về suy luận pháp lý tiếng Việt và cấu trúc lập luận tam đoạn luận (Syllogism), được tối ưu hóa qua phương pháp GRPO (Group Relative Policy Optimization).
*   **Trạng thái phần cứng:** Đóng băng hoàn toàn (Frozen), tải ở định dạng **4-bit NF4** để chạy forward-pass xuất logits nhằm tiết kiệm VRAM.
*   **Cơ chế hoạt động:**
    *   **Input (Đầu vào):** 
        *   *Trong Phase 1 (Offline KD):* Nhận chuỗi Prompt chuẩn chứa ngữ cảnh pháp lý và câu hỏi gốc $x = (c, q)$.
        *   *Trong Phase 2 (On-Policy KD - Oracle Mode):* Nhận Prompt kèm theo **Thông tin Đặc quyền (Privileged Information)** là đáp án đúng $y^*$ của tập dữ liệu huấn luyện để hướng dẫn mô hình học sinh tốt nhất.
    *   **Output (Đầu ra):** Phân phối xác suất (Soft logits) tại mỗi bước sinh từ (ở Phase 1 tính toán trên Top-50 tokens có xác suất cao nhất) làm mục tiêu chưng cất cho Student.

---

## 2. Student Models (Mô hình Học sinh)

### 2.1. Phân khúc 0.6B Parameters
*   **Tên mô hình gốc:** `HoangVuSnape/qwen3-0.6b-sft` (Qwen3 0.6B đã được huấn luyện SFT trên dữ liệu pháp lý).
*   **Định dạng chung:**
    *   **Input (Đầu vào):** Prompt chứa ngữ cảnh và câu hỏi pháp lý định dạng theo cấu trúc Alpaca.
    *   **Output (Đầu ra):** Chuỗi suy nghĩ Chain-of-Thought (CoT) nằm trong cặp thẻ `<think> ... </think>` và câu trả lời cuối cùng (Ví dụ: "Có liên quan"/"Không liên quan" cho NLI; "A/B/C/D" cho MCQ; lập luận suy diễn cho Syllogism).
*   **Các phiên bản huấn luyện cụ thể:**

| Phiên bản Run | Phương pháp PEFT | Siêu tham số (LR & Alpha) | Đặc trưng Input & Output |
| :--- | :--- | :--- | :--- |
| **E1_offline_kd** | LoRA V1 (16-bit) | LR: `1e-4`, Alpha: `0.5`, Temp: `2.0` | **Input:** Offline static data.<br>**Output:** Bị ảnh hưởng bởi Catastrophic Forgetting do LR quá cao. |
| **A1_offline_kd** | LoRA V2 (16-bit) | LR: `2e-5`, Alpha: `0.3`, Temp: `1.5` | **Input:** Offline static data.<br>**Output:** Hội tụ tốt hơn nhờ giảm LR và Alpha. |
| **E2_online_kd** | QLoRA V1 (4-bit NF4) | LR: `1e-4`, Alpha: `0.5`, Temp: `2.0` | **Input:** Offline static data.<br>**Output:** Gặp lỗi suy giảm logic nhẹ. |
| **A2_offline_kd** | QLoRA V2 (4-bit NF4) | LR: `2e-5`, Alpha: `0.3`, Temp: `1.5` | **Input:** Offline static data.<br>**Output:** Checkpoint tốt nhất của bản 0.6B Offline KD. |
| **A2_onpolicy_onpolicy_kd_multigpu** | QLoRA + On-Policy KD | LR: `2e-5`, Alpha: `0.5`, Temp: `2.0` | **Input:** Prompt gốc.<br>**Output:** Tự sinh quỹ đạo trả lời (rollouts), học sửa sai từ Teacher. |
| **A2_dpo** | QLoRA + DPO | LR: `5e-6`, Beta: `0.1` | **Input:** Cặp Preference (Chosen/Rejected).<br>**Output:** Căn chỉnh cấu trúc suy nghĩ dựa trên lỗi sai chẩn đoán. |

---

### 2.2. Phân khúc 1.7B Parameters
*   **Tên mô hình gốc:** `HoangVuSnape/qwen3-1.7b-sft` (Qwen3 1.7B đã được huấn luyện SFT trên dữ liệu pháp lý).
*   **Định dạng chung:**
    *   **Input (Đầu vào):** Tương tự bản 0.6B (Prompt Alpaca chứa ngữ cảnh và câu hỏi pháp lý).
    *   **Output (Đầu ra):** Chuỗi suy nghĩ và câu trả lời cuối cùng với chất lượng ngữ nghĩa và độ bám sát logic cao hơn đáng kể so với bản 0.6B.
*   **Các phiên bản huấn luyện cụ thể:**

| Phiên bản Run | Phương pháp PEFT | Siêu tham số (LR & Alpha) | Đặc trưng Input & Output |
| :--- | :--- | :--- | :--- |
| **E3_online_kd** | LoRA V1 (16-bit) | LR: `1e-4`, Alpha: `0.5`, Temp: `2.0` | **Input:** Offline static data.<br>**Output:** Bị ảnh hưởng bởi Catastrophic Forgetting do LR quá cao. |
| **A3_offline_kd** | LoRA V2 (16-bit) | LR: `2e-5`, Alpha: `0.3`, Temp: `1.5` | **Input:** Offline static data.<br>**Output:** Checkpoint tốt nhất của bản 1.7B Offline KD. |
| **A3_onpolicy_onpolicy_kd** | LoRA + On-Policy KD | LR: `2e-5`, Alpha: `0.5`, Temp: `2.0` | **Input:** Prompt gốc.<br>**Output:** Tự sinh quỹ đạo trả lời, học sửa sai từ Teacher. |
| **A3_dpo** | LoRA + DPO | LR: `5e-6`, Beta: `0.1` | **Input:** Cặp Preference (Chosen/Rejected).<br>**Output:** Căn chỉnh tối ưu hóa logic suy luận pháp lý. |

---

## 3. LLM Judge Model (Mô hình Giám khảo)

*   **Tên mô hình:** `gpt-4o-mini`
*   **Vai trò:** Đánh giá ngữ nghĩa chất lượng đầu ra của Student trong Phase 3 để chuẩn bị cặp dữ liệu huấn luyện DPO.
*   **Input (Đầu vào):** 
    1.  Câu trả lời thực tế do Student tự sinh ra (`student_output`).
    2.  Nhãn câu trả lời chuẩn (`ground_truth`).
    3.  Hệ thống tiêu chí chấm điểm (Rubric) chi tiết về tính đúng đắn của đáp án và logic suy nghĩ.
*   **Output (Đầu ra):** Phản hồi phân loại chẩn đoán lỗi logic thành một trong 4 nhãn:
    *   `OK` (Đúng cả suy nghĩ lẫn đáp án)
    *   `RISKY` (Đúng đáp án nhưng suy nghĩ bị ảo giác/sai lệch)
    *   `PARTIAL` (Đi đúng hướng suy nghĩ nhưng kết luận sai)
    *   `WRONG` (Sai hoàn toàn)
