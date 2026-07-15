# DANH SÁCH CÂU HỎI VẤN ĐÁP THEO TỪNG PHẦN
### Dự đoán câu hỏi hội đồng + gợi ý hướng trả lời
*(Ký hiệu: ⭐ = câu dễ bị hỏi nhất · 🔥 = câu "đâm" khó · 💡 = mẹo phòng thủ)*

---

## PHẦN 1 — Mở đầu (Bài toán, động lực, mục tiêu)

⭐ **Vì sao chọn SLM nhỏ mà không dùng thẳng LLM lớn?**
→ 3 lý do: riêng tư dữ liệu (Luật An ninh mạng — hồ sơ không rời cơ quan), chi phí API, độ trễ/khả dụng. SLM on-premise giải quyết cả 3.

**"Nhỏ" là bao nhiêu? Vì sao 0.6B và 1.7B?**
→ <3B để chạy trên T4 16GB. Chọn 2 mốc để **khảo sát ảnh hưởng của quy mô** (đây là một phát hiện chính, không phải tùy tiện).

🔥 **Đề tài giải quyết vấn đề gì mà công trình trước chưa làm?**
→ Khoảng trống: (1) chưng cất offline chuẩn bỏ qua exposure bias; (2) dữ liệu preference lọc bằng luật cứng gây nhiễu; (3) chưa ai khảo sát các phương pháp này ở quy mô <2B cho **tiếng Việt**.

🔥 **Teacher Qwen3-4B-legal-GRPO ở đâu ra? Có phải đóng góp của em không?**
→ **Không** — teacher được kế thừa như checkpoint đã căn chỉnh sẵn từ pha nghiên cứu trước (thangvip). Huấn luyện teacher **ngoài phạm vi** đề tài. 💡 Nói rõ ranh giới này để tránh bị quy "nhận vơ".

**Đóng góp lớn nhất của em là gì (1 câu)?**
→ Chứng minh pipeline chưng cất đa pha khả thi cho SLM pháp lý tiếng Việt, **và** chỉ ra ngưỡng dung lượng tối thiểu để alignment nâng cao có lợi (0.6B sụp đổ, 1.7B hưởng lợi).

---

## PHẦN 2 — Kiến thức nền (Background)

⭐ **Knowledge Distillation là gì? Vì sao học "soft label" tốt hơn "hard label"?**
→ Student bắt chước **toàn bộ phân phối xác suất mềm** của teacher (dark knowledge — quan hệ giữa các token), không chỉ đáp án cuối. Soft label mang tín hiệu sửa lỗi theo tác vụ mà hard label không có.

**Phân biệt LoRA và QLoRA?**
→ LoRA: chèn ma trận low-rank, base vẫn trainable qua adapter → dễ quên. QLoRA: đông cứng base ở **4-bit NF4**, chỉ cập nhật adapter → điều chuẩn ngầm, chống quên.

🔥 **Exposure bias là gì? Vì sao nghiêm trọng trong pháp lý?**
→ Khi train chỉ thấy tiền tố đúng (teacher-forcing), nhưng inference phải nối tiếp chuỗi **tự sinh**. Một lỗi nhỏ ở bước $t$ lan truyền, tích lũy → sụp đổ chuỗi lập luận. Trong luật, một bước sai làm hỏng cả phán quyết.

**DPO khác RLHF/PPO chỗ nào?**
→ DPO tối ưu trực tiếp trên cặp (chosen, rejected) không cần reward model tường minh, ổn định và rẻ hơn PPO.

🔥 **GRPO là gì (teacher dùng)?** → Group Relative Policy Optimization — biến thể RL căn chỉnh, so sánh tương đối trong nhóm rollout. 💡 Nếu bị hỏi sâu: đây là của teacher (ngoài phạm vi), em nắm khái niệm nhưng không tự huấn luyện.

---

## PHẦN 3 — Phương pháp (Pipeline 3 pha)

⭐🔥 **Vẽ/giải thích lại pipeline. Nó là chuỗi tuyến tính 1→2→3 phải không?**
→ **KHÔNG.** Từ base SFT chung, **2 nhánh độc lập**: nhánh offline (Pha 1 → Pha 3 DPO trên ckpt Pha 1) và nhánh on-policy (Pha 2 áp thẳng lên SFT). Pha 2 **không** khởi từ Pha 1. Reference của DPO = checkpoint Pha 1. 💡 Đây là điểm rất dễ bị bắt lỗi — nhớ kỹ.

**Pha 1: hàm mất mát? Vì sao α=0.3 (nghiêng về CE)?**
→ $\mathcal{L}=\alpha\mathcal{L}_{KD}+(1-\alpha)\mathcal{L}_{CE}$. α=0.3 → 70% CE làm "neo" giữ đáp án đúng, tránh chạy theo nhiễu của teacher. Top-K=50 để tiết kiệm lưu trữ/tính toán logits.

⭐🔥 **Pha 2: teacher được xem đáp án vàng $y^*$ — đây chẳng phải gian lận / rò rỉ đáp án sao?**
→ Thừa nhận thẳng đây là **rủi ro rò rỉ**. Hai cơ chế hạn chế: (1) tín hiệu KD áp trên **rollout tự sinh** của student chứ không phải chuỗi vàng; (2) **KL clipping C=5.0** chặn đóng góp của token tự tin cao (kể cả token đáp án). $y^*$ đóng vai **điều kiện dẫn dắt** cho teacher nối tiếp, không phải mục tiêu để chép. 💡 Và đã liệt kê rõ trong Giới hạn: **chưa ablation** phần học thật vs rò rỉ.

**Vì sao cần KL clipping?** → Token định dạng (`<think>`, `<major_premise>`...) entropy cực thấp → KL bùng nổ, lấn át gradient, ép student học **định dạng** thay vì **nội dung pháp lý**. Clipping chặn trần penalty mỗi token.

**Pha 3: cặp preference tạo thế nào?** → Chosen = đáp án vàng; Rejected = rollout student bị LLM Judge chẩn đoán sai. β=0.1, LR 5e-6, ref = ckpt Pha 1.

🔥 **Vì sao dùng LLM Judge (GPT-4o-mini) thay vì luật cứng?**
→ Luật cứng (regex + ROUGE-L) thưởng "đoán may" (MCQ đúng nhưng lập luận sai) và loại đáp án đúng-nhưng-khác-định-dạng. Judge đánh giá **ngữ nghĩa** cả `<think>` lẫn kết luận → tập preference sạch hơn (số liệu ablation 2).

---

## PHẦN 4 — Thực nghiệm & Kết quả

⭐ **Dữ liệu bao nhiêu? Chia thế nào?**
→ ViLawQA gộp 3,428 → phân tầng theo `task_type`: 2,603 train / 322 val / 325 test (Split). VLSP2025 Public-Test 440 mẫu = OOD.

⭐🔥 **"Overall" tính thế nào? Trộn accuracy với ROUGE-L có hợp lệ không?**
→ Trung bình **không trọng số** 3 metric. Thừa nhận đây là chỉ báo **thô** (khác thang đo) → báo cáo đã nêu rõ và **phân tích chính dựa trên per-task**, không dựa vào Overall. 💡 Câu này gần như chắc chắn bị hỏi.

⭐🔥 **Vì sao Syllogism dùng ROUGE-L mà không dùng judge chính thức của VLSP?**
→ Để **so sánh nhất quán** giữa 8 mô hình bằng một metric nội bộ, deterministic. Đã ghi chú rõ đây là **proxy**, không phải điểm thi chính thức.

🔥 **Kết quả cho thấy distillation nhiều khi TỆ hơn SFT (OOD 0.6395 > mọi bản distilled). Vậy đề tài có ý nghĩa gì?**
→ Đây là **phát hiện trung thực về đánh đổi chuyên biệt hóa vs tổng quát hóa**, không phải thất bại. In-distribution: distilled 1.7B vượt SFT (NLI 0.8156, Syl 0.5684). On-policy KD 1.7B vẫn là **distilled tốt nhất OOD** (0.6155, NLI 0.920) → phá exposure bias giúp tổng quát hóa. Điểm yếu tập trung ở **MCQ**.

🔥 **Vì sao MCQ của mọi mô hình distilled đều kém (<0.56 vs 0.74)?**
→ Chưng cất tối ưu chuỗi suy luận `<think>` (NLI/Syllogism) làm dịch chuyển hành vi khỏi chọn A/B/C/D; DPO còn đè MCQ mạnh hơn (1.7B DPO MCQ tụt 0.4455→0.20). Đây là biểu hiện rõ nhất của chuyên biệt hóa.

⭐🔥 **Vì sao 0.6B sụp đổ khi on-policy/DPO (0.46→0.31→0.19)?**
→ Dung lượng dưới 1 tỷ tham số không hấp thụ nổi: rollout tự sinh quá nhiễu + hình phạt probability-displacement của DPO quá gắt → mất ổn định. Kết luận: **alignment nâng cao chỉ có lợi trên ngưỡng dung lượng tối thiểu**.

**QLoRA thật sự chống quên hay chỉ ngẫu nhiên?**
→ Bằng chứng động lực học: CE loss 0.64→0.37 (QLoRA, giảm đều) vs 1.00→0.56 (LoRA, cao & trôi lên). CE loss là proxy cho việc giữ tri thức SFT. Cùng cấu hình V2 → cô lập đúng ảnh hưởng của adapter.

🔥 **A3 offline (0.5322) chỉ hơn A3 DPO (0.5280) một chút — sao dám kết luận?**
→ **Không** kết luận DPO cải thiện Overall — báo cáo nói rõ chênh lệch này **có thể trong khoảng nhiễu** (chạy một lần, n=325). DPO thắng ở **per-task NLI & Syllogism**, đó mới là điều được khẳng định.

**V1 vs V2 khác gì?** → V1 (LR 1e-4, batch 16, α=0.5) gây catastrophic forgetting (CE 0.3→0.7, sinh văn bản trộn Việt-Anh-Trung). V2 (LR 2e-5, batch 64, α=0.3) ổn định (CE 0.5→0.25).

---

## PHẦN 5 — Kết luận, Giới hạn, Hướng phát triển

⭐ **Giới hạn lớn nhất của đề tài?**
→ Nêu theo 4 mảng: **mô hình** (0.6B sụp đổ, chưng cất chuyên biệt hóa), **đo lường** (metric thô, chạy một lần), **phương pháp** (rò rỉ Oracle chưa ablation), **nguồn lực** (phụ thuộc API ngoài, phần cứng). 💡 Chủ động nêu trước để thể hiện nghiên cứu trung thực.

🔥 **Cam kết "bảo mật/on-premise" có mâu thuẫn khi Pha 3 gọi API GPT-4o-mini không?**
→ Có giới hạn: cam kết riêng tư chỉ áp dụng cho **inference & triển khai on-premise**, **không** cho giai đoạn huấn luyện (tạo dữ liệu preference). Đã ghi rõ trong Giới hạn. Hướng khắc phục: thay bằng judge mã nguồn mở self-host.

🔥 **Kết quả chỉ chạy một lần thì độ tin cậy thế nào?**
→ Thừa nhận: không nhiều seed, không khoảng tin cậy, n=325 → các chênh lệch nhỏ đọc như **xu hướng**, không khẳng định thống kê. Đây là giới hạn đã liệt kê.

**Nếu có thêm tài nguyên, làm gì tiếp?**
→ (1) continual learning (replay/EWC/L2-SP/KL anchor) chống quên; (2) alignment nhạy dung lượng (KTO/ORPO/IPO, curriculum) cứu 0.6B; (3) trộn dữ liệu tổng quát chống chuyên biệt hóa; (4) mở rộng student ≥4B, đa teacher/self-distillation, đa lĩnh vực luật.

---

## CÂU HỎI TỔNG QUÁT / "BẪY" THƯỜNG GẶP

- 🔥 **"Tính mới thực sự (novelty) là gì?"** → Không phải phát minh KD/DPO, mà là **tích hợp 3 pha + chẩn đoán ngữ nghĩa + khảo sát phân tầng quy mô** cho pháp lý **tiếng Việt**, kèm phát hiện ngưỡng dung lượng.
- 🔥 **"Kết quả tốt nhất chỉ ~0.53 Overall — mô hình đã dùng được thực tế chưa?"** → Chưa phải sản phẩm cuối; đây là **nghiên cứu khả thi (feasibility)** + đặc tả hành vi. Trung thực về khoảng cách tới triển khai.
- **"Vì sao 0.6B dùng QLoRA còn 1.7B dùng LoRA?"** → 0.6B nhạy quên hơn nên cần QLoRA; 1.7B đủ dung lượng chịu LoRA. (Nếu bị đẩy: nhất quán hóa adapter là một điểm có thể cải thiện.)
- **"Em tự làm phần nào?"** → Toàn bộ pipeline 3 pha, dữ liệu, thực nghiệm, đánh giá, ablation. Teacher & GRPO là kế thừa (ngoài phạm vi).
- 💡 **Nguyên tắc trả lời:** khi bị "đâm" vào một giới hạn → **thừa nhận thẳng** → chỉ vào chỗ báo cáo đã nêu → nói hướng khắc phục. Đừng phòng thủ hay chối.
