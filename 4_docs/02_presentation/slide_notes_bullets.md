# Speaker Notes (Cue Cards) — CD1  ·  bản 25 slide
### "A Hybrid Knowledge Distillation and Preference Optimization Framework…"
**SV:** Hoàng Đình Quý Vũ — 252805008 · **GVHD:** PGS. TS. Lê Anh Cường

> Quy ước: mỗi **gạch đầu dòng ≈ 1 dòng note (~8–12 từ)**, nói thành câu khi trình bày.
> Notes khớp đúng thứ tự 25 slide trong file `VN_Legal_AI_CD1.pptx` hiện tại.
> ⚠️ Lưu ý: **Slide 13** = bảng kết quả tổng hợp đầy đủ (Table 4.6, dán dạng ẢNH) — đây là slide kết quả chính. **Slide 21** = bản rút gọn SFT-vs-Distilled (chữ-native), đang nằm trong khu appendix → có thể bỏ hoặc giữ làm phương án nói nhanh. Số trang footer "103/16" ở slide 13 là lỗi đánh số, nên sửa thành 13.

---

## Slide 1 — Title *(~20s)*
- Kính chào quý thầy cô trong hội đồng.
- Em là Hoàng Đình Quý Vũ, Khoa CNTT, ĐH Tôn Đức Thắng.
- Đề tài: khung lai chưng cất tri thức + tối ưu sở thích.
- Cho hỏi-đáp pháp lý tiếng Việt trên mô hình nhỏ.
- Dưới hướng dẫn của thầy Lê Anh Cường.

## Slide 2 — Table of Contents *(~20s)*
- Bài gồm bốn mạch chính.
- Một: đặt vấn đề và mục tiêu.
- Hai: kiến trúc pipeline và ba pha.
- Ba: thiết lập thí nghiệm.
- Bốn: kết quả, giới hạn, kết luận.
- Em dừng lâu nhất ở phần kết quả.

## Slide 3 — Problem & Motivation *(~55s)*
- SLM dưới 2 tỷ tham số hấp dẫn cho pháp lý.
- Chạy tại chỗ, rẻ, độ trễ thấp.
- Quan trọng nhất: hồ sơ vụ án không rời tổ chức.
- Nhưng mô hình nhỏ khó suy luận nhiều bước.
- Khó khăn 1: khoảng cách dung lượng so với thầy lớn.
- Khó khăn 2: exposure bias — lỗi sớm tích lũy.
- Hướng đi: chưng cất thầy 4B-GRPO vào trò nhỏ.

## Slide 4 — Research Gap & Objectives *(~50s)*
- Có ba khoảng trống.
- Một: offline KD tiêu chuẩn bỏ qua exposure bias.
- Hai: lọc preference bằng luật cứng dễ vỡ.
- Ba: ít nghiên cứu ở quy mô dưới 2B tiếng Việt.
- Năm mục tiêu: khung lai; QLoRA điều chuẩn ẩn.
- On-policy giảm exposure bias; DPO dẫn dắt bằng chẩn đoán.
- Đánh giá in-distribution lẫn out-of-distribution.

## Slide 5 — Proposed Pipeline *(~60s)*
- Nhấn mạnh: KHÔNG phải chuỗi tuyến tính ba pha.
- Gốc chung duy nhất là mô hình SFT.
- Từ SFT rẽ hai nhánh độc lập, song song.
- Nhánh trên: On-Policy KD (Pha 2) — đi thẳng từ SFT.
- Nhánh dưới: Pha 1 offline rồi nối tiếp Pha 3 DPO.
- Pha 2 từ SFT; Pha 3 nối + tham chiếu Pha 1.
- Thầy là Qwen3-4B Vietnamese-legal-GRPO.

## Slide 6 — Phase 1 · Offline Logit KD (sơ đồ) *(~40s)*
- Trò bắt chước phân phối mềm của thầy.
- Trên một tập dữ liệu tĩnh.
- Cùng input vào thầy (đóng băng, Top-50 logit) và trò (QLoRA).
- Phát hiện chính: QLoRA là bộ điều chuẩn ẩn.
- Đóng băng base 4-bit, chỉ cập nhật adapter.
- Ngăn quên thảm họa ở quy mô nhỏ.

## Slide 7 — Phase 1 · Công thức *(~45s)*
- Tổng loss = alpha·L_KD + (1−alpha)·L_CE, alpha 0.3.
- 70% trọng số đặt vào neo cross-entropy.
- L_CE chỉ tính trên token đầu ra (prompt mask −100).
- Soft target chuẩn hóa trên Top-50, chia nhiệt độ T.
- L_KD là KL giữa thầy–trò, nhân T-bình-phương.
- Giữ độ lớn gradient khi nhiệt độ cao.

## Slide 8 — Phase 2 · On-Policy KD (sơ đồ) *(~45s)*
- Pha 2 giải quyết exposure bias.
- Trò tự sinh rollout (nhiệt độ 0.7, top-p 0.9).
- Thầy ở chế độ Oracle — thấy đáp án vàng y-sao.
- Thầy chấm sửa lỗi ở cấp token.
- KL có cắt ngưỡng, chống nổ gradient.
- Huấn luyện trên chính quỹ đạo tự sinh.

## Slide 9 — Phase 2 · Công thức *(~45s)*
- Tổng loss cân bằng on-policy KD và neo SFT, alpha 0.5.
- KL từng token tính trên toàn từ vựng.
- Tính trên chuỗi do trò tự sinh.
- KL clip = min(KL, C) với C = 5.0.
- Cắt token cấu trúc <think> có KL khổng lồ.
- Loss = T-bình-phương nhân KL-clip trung bình.

## Slide 10 — Phase 3 · Diagnosis-Driven DPO (sơ đồ) *(~45s)*
- Pha 3 sửa các lỗi suy luận còn sót.
- Trò sinh rollout, LLM Judge GPT-4o-mini kiểm toán.
- Kiểm cả chuỗi suy nghĩ lẫn kết luận cuối.
- OK thì bỏ; có lỗi thì tạo cặp sở thích.
- Chọn = đáp án vàng; loại = lỗi của trò.
- Chẩn đoán ngữ nghĩa sạch hơn lọc luật.

## Slide 11 — Phase 3 · Công thức *(~45s)*
- DPO loss trên bộ ba (x, y_w, y_l), beta 0.1.
- y_w là chosen (vàng), y_l là rejected (lỗi).
- Tăng xác suất y_w, giảm y_l.
- Reward ngầm = beta·log tỉ số chính sách / tham chiếu.
- Margin tăng ⇒ tách biệt sở thích tốt trên train.
- Tham chiếu là checkpoint Pha 1 (đóng băng).

## Slide 12 — Experimental Setup & Data *(~45s)*
- Dữ liệu ViLawQA từ tài khoản thangvip.
- Ba tác vụ: MCQ, NLI ba lớp, Syllogism.
- Train 2.603 · valid 322 · test nội bộ 325.
- VLSP2025 Public-Test 440 mẫu — benchmark OOD.
- Thầy Qwen3-4B; trò 0.6B và 1.7B.
- Overall chỉ là trung bình thô.
- Điểm từng tác vụ mới là cơ sở chính.

## Slide 13 — Results · Final Evaluation (BẢNG CHÍNH) *(~75s — DỪNG LÂU)*
- Đây là bảng kết quả tổng hợp, phần em lưu ý nhất.
- Gồm cả in-distribution (Split-Test) và OOD (VLSP2025).
- Nhấn mạnh: per-task là cơ sở chính, Overall chỉ tham khảo.
- In-dist: 1.7B DPO đạt NLI cao nhất 0.8156, Syllogism 0.5684.
- 1.7B Offline KD có Overall cao nhất 0.5322.
- 0.6B sụp đổ: Overall 0.4622 → 0.3104 → 0.1861.
- OOD: SFT 1.7B tổng quát hóa tốt nhất (Overall 0.6395, MCQ 0.740).
- On-Policy 1.7B giữ NLI cao nhất 0.920, Syllogism R-L 0.372.
- Thông điệp: đánh đổi chuyên biệt vs tổng quát.

## Slide 14 — Limitations *(~45s)*
- Em chủ động nêu các giới hạn.
- 0.6B bất ổn: on-policy/DPO phản tác dụng.
- Chưng cất chuyên biệt hóa quá mức trên OOD.
- Syllogism dùng ROUGE-L proxy; Overall trộn thang đo.
- Đánh giá một lần chạy, n=325 — khe hẹp có thể nhiễu.
- Rò rỉ đáp án từ Oracle ở Pha 2 (chưa ablation).
- Phụ thuộc API ngoài và phần cứng giới hạn.

## Slide 15 — Conclusion & Contributions *(~45s)*
- Bốn đóng góp chính.
- Một: khung lai phân nhánh từ SFT chung.
- Hai: lượng tử hóa thiết yếu — QLoRA điều chuẩn ẩn.
- CE giảm 0.64→0.37 (QLoRA) vs 1.00→0.56 (LoRA).
- Ba: exposure bias có ý nghĩa — on-policy thắng OOD.
- Bốn: chẩn đoán ngữ nghĩa cho DPO vượt lọc luật.
- Mô hình 1.7B suy luận pháp lý mạnh in-domain.

## Slide 16 — Thank You *(~10s)*
- Em xin chân thành cảm ơn quý thầy cô.
- Rất mong nhận câu hỏi và góp ý từ hội đồng.

---
# PHỤ LỤC (lật khi được hỏi)

## Slide 17 — Appendix · Key Hyperparameters
- Offline KD V2: LR 2e-5, alpha 0.3, T 1.5, Top-K 50.
- QLoRA: 4-bit NF4, rank 16, alpha 32, dropout 0.05.
- On-Policy: decode T 0.7, top-p 0.9, KL clip bật.
- On-policy chậm gấp ~8 lần offline.

## Slide 18 — Appendix · Algorithm 1 (Phase 1 pseudocode)
- Logit thầy cache trước (Top-50) ⇒ offline, không forward thầy.
- Trò teacher-forced trên y-sao.
- Loss = T²·KL + neo CE; chỉ cập nhật adapter LoRA.
- Trả về checkpoint Pha 1 (gốc cho Pha 3).

## Slide 19 — Appendix · Algorithm 2 (Phase 2 pseudocode)
- Trò tự sinh rollout; lặp từng token phản hồi.
- Thầy Oracle điều kiện trên rollout + đáp án vàng.
- KL từng token, cắt tại C = 5.0.
- Loss on-policy + neo SFT; trả về On-Policy SLM.

## Slide 20 — Appendix · Algorithm 3 (Phase 3 pseudocode)
- Giai đoạn A: LLM Judge gán OK/RISKY/PARTIAL/WRONG.
- Chỉ RISKY/WRONG thành cặp (vàng vs lỗi).
- Lấy mẫu phân tầng cân bằng tác vụ.
- Giai đoạn B: tối ưu DPO; trả về DPO-Aligned SLM.

## Slide 21 — Results (bản rút gọn SFT-vs-Distilled) — dùng khi cần nói nhanh
- Bản tóm tắt: chưng cất 1.7B tốt nhất so với SFT.
- NLI 0.745→0.816, Syllogism 0.207→0.568, MCQ 0.500→0.446.
- Overall 0.484→0.532; suy luận tăng, MCQ giảm.
- (Trùng ý Slide 13 — giữ làm phương án nói nhanh, hoặc bỏ.)

## Slide 22 — Appendix · Dataset & HP Tuning
- Bảng 4.1: phân bố ba tác vụ qua các split.
- Valid 322 sau làm sạch (so với 343 lý thuyết).
- Bảng 4.2: V1 gây quên thảm họa, V2 ổn định lại.

## Slide 23 — Appendix · Ablations & Qualitative
- Bảng 4.5: QLoRA CE 0.64→0.37 vs LoRA 1.00→0.56.
- QLoRA luôn thấp hơn ⇒ giữ kiến thức tốt hơn.
- Bảng 4.6: Judge nâng OK 11→21%, WRONG 41→64%.
- Tập preference sạch hơn cho DPO.

## Slide 24 — Appendix · Anticipated Questions (1/2)
- DPO Overall thấp hơn offline → Overall thô, per-task chính.
- DPO tối đa hai tác vụ suy luận (NLI, Syllogism).
- Khe 0.004 nằm trong nhiễu một-lần-chạy.

## Slide 25 — Appendix · Anticipated Questions (2/2)
- Dùng GPT-4o-mini → bảo mật phạm vi suy luận.
- Mô hình triển khai chạy hoàn toàn tại chỗ.
- API chỉ chạm dựng dữ liệu một lần, không chạm hồ sơ sống.
