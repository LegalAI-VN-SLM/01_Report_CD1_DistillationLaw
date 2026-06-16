# Overview Summary: Multi-Phase Distillation & Alignment for Vietnamese Legal SLMs

Báo cáo này tổng hợp toàn bộ kết quả thực nghiệm từ 10 runs huấn luyện, so sánh sự đánh đổi hiệu năng và tài nguyên GPU qua **3 Hướng chiến lược (3 Phases)** của dự án.

---

## ⚖️ 1. So sánh Hiệu năng giữa các Phase chính

Dựa trên bảng số liệu tổng hợp, chúng ta rút ra các kết luận thực nghiệm quan trọng:

### Phase 1 (Offline KD V2) vs SFT Baseline:
* Chưng cất logit tĩnh giúp Student hấp thụ tri thức nền của Teacher một cách hiệu quả, đặc biệt cải thiện mạnh ở tác vụ **MC Accuracy** (Trắc nghiệm). 
* Bản 1.7B KD đạt điểm MCQ vượt trội và sửa được nhiều lỗi chọn đáp án sai của bản SFT gốc. Tuy nhiên, nó vẫn gặp khó khăn ở tác vụ tự sinh lập luận dài (Syllogism).

### Phase 2 (On-Policy KD) vs Phase 1 (Offline KD):
* **Ưu điểm:** On-Policy giải quyết Exposure Bias bằng cách huấn luyện Student trên các chuỗi tự sinh, đẩy khả năng logic bắc cầu (**NLI Accuracy**) của bản 1.7B lên mức kỷ lục là **$0.7943$** (so với $0.7305$ của Offline).
* **Nhược điểm:** Làm giảm nhẹ khả năng sinh văn bản lập luận tự do (Syllogism Rouge-L bị phạt giảm do KL Divergence ép mô hình Student quá chặt theo phân phối tự sinh) và làm giảm nhẹ MC Accuracy.
* **Chi phí cực cao:** Tốn VRAM gấp đôi ở bản 1.7B ($45.2\text{ GB}$ vs $22.18\text{ GB}$) và chạy chậm hơn gấp 8 lần.

### Phase 3 (DPO Alignment):
* Đóng vai trò là chặng tinh chỉnh hành vi cuối cùng. DPO không thay đổi tri thức nền (MC/NLI) mà tập trung định hình chuẩn xác **cấu trúc suy nghĩ (Thinking Block)** và loại bỏ các lỗi logic nhờ dữ liệu preference được gán nhãn chẩn đoán chất lượng từ LLM Judge.
* **Lưu ý đặc thù về nguồn dữ liệu & base model:** Dữ liệu preference được gán nhãn của DPO được tạo dựng từ các lỗi sai của Student 0.6B QLoRA chưng cất offline (`A2_offline_kd`), sau đó dùng chung để căn chỉnh DPO cho cả 0.6B và 1.7B. Base checkpoint để huấn luyện DPO được lấy trực tiếp sau chặng **Offline KD V2** chứ chưa phải từ chặng Online/On-Policy KD.

---

## ⚡ 2. So sánh Chi phí Tài nguyên & Thời gian Huấn luyện

Biểu đồ hiệu năng phần cứng chỉ ra sự đánh đổi phần cứng rõ rệt:
* **Offline KD V2 (A2, A3) là tối ưu nhất về tài nguyên:** Bản 0.6B QLoRA chỉ tốn $19.65\text{ GB}$ Peak VRAM và mất $1.39\text{ giờ}$ huấn luyện. Bản 1.7B LoRA chỉ tốn $22.18\text{ GB}$ Peak VRAM và mất $1.32\text{ giờ}$ huấn luyện trên Kaggle T4.
* **On-Policy KD cực kỳ đắt đỏ:** Do phải duy trì hai mô hình (Student tự sinh câu trả lời + Teacher tính logit trên quỹ đạo của Student) song song, thời gian huấn luyện bị kéo dài khủng khiếp (bản 0.6B chạy mất **$10.88\text{ giờ}$**). Bản 1.7B đòi hỏi lượng bộ nhớ khổng lồ ($45.2\text{ GB}$ VRAM), vượt quá giới hạn GPU phổ thông.

---

## 🎯 3. Đề xuất Chiến lược tối ưu hóa cho Đồ án/Báo cáo

Để xây dựng một mô hình SLM tối ưu cho miền luật Việt Nam với chi phí phần cứng hợp lý nhất, quy trình kết hợp đề xuất là:

```
Step 1: QLoRA SFT Baseline (Thiết lập nền móng biểu diễn ngôn ngữ luật)
                 │
                 ▼
Step 2: Offline Logit KD V2 (Chưng cất tri thức thô ổn định, chi phí thấp, tăng mạnh MCQ)
                 │
                 ▼
Step 3: DPO Preference Alignment (Căn chỉnh cấu trúc suy nghĩ bằng LLM Judge, loại bỏ lỗi logic)
```

Sự kết hợp này giúp tận dụng tối đa thế mạnh của chưng cất tri thức tĩnh (ổn định, chạy nhanh) và căn chỉnh ưu tiên DPO (lọc sạch hành vi suy nghĩ, tinh gọn định dạng) mà không phải gánh chịu chi phí huấn luyện khổng lồ của On-Policy KD.

---

## 📂 Danh sách các biểu đồ PNG tham chiếu (Folder `chart/`):
1. **`fig1_phase1_v1_vs_v2_comparison.png`:** So sánh hiệu quả cấu hình V1 vs V2.
2. **`fig2_phase1_ce_loss_trend.png`:** Xu hướng CE Loss chứng minh hiện tượng catastrophic forgetting vs regularization.
3. **`fig3_phase2_onpolicy_vs_offline.png`:** So sánh hiệu năng On-Policy KD vs Offline KD.
4. **`fig4_phase3_llm_judge_distribution.png`:** Phân phối chẩn đoán của LLM Judge (A2 vs A3).
5. **`fig5_phase3_dpo_reward_margins.png`:** Tăng trưởng Reward Margin của DPO.
6. **`fig6_hardware_efficiency_3_phases.png`:** So sánh Peak VRAM và Runtime qua các chặng.

---
* **Tài liệu tham chiếu cấu hình mô hình chi tiết:** [model_config.md](file:///e:/DoCode/1%20VN-Legal-AI/legal-slm-finetune/1_download_wandb_evaluate/Analyze_final/md_report/model_config.md)
* **Tài liệu phân tích độ tin cậy và tính thống nhất của đánh giá:** [eval_analysis.md](file:///e:/DoCode/1%20VN-Legal-AI/legal-slm-finetune/1_download_wandb_evaluate/Analyze_final/md_report/eval_analysis.md)
