# TÓM TẮT BÁO CÁO (bản SÂU — theo từng mục)
## Chưng cất tri thức cho suy luận pháp luật tiếng Việt trên các Mô hình Ngôn ngữ Nhỏ
*Khung lai: Offline Logit KD · On-Policy KD · Diagnosis-Driven DPO — Qwen3-0.6B / 1.7B, neo bằng QLoRA*

> **Abstract (1 đoạn):** Triển khai LLM thương mại trong pháp lý bị chặn bởi quyền riêng tư, chi phí, độ trễ; còn SLM nhỏ gọn lại yếu suy luận đa bước. Đề tài đề xuất **khung lai chưng cất tri thức + tối ưu hóa sở thích**, chuyển khả năng suy luận từ một **teacher đã căn chỉnh bằng RL (GRPO)** vào SLM pháp lý tiếng Việt ở quy mô **0.6B và 1.7B**. Từ một base **SFT** chung, khung rẽ **hai nhánh song song**: nhánh **on-policy (Pha 2)** giảm exposure bias qua rollout của học sinh, và nhánh **offline** kết hợp **Offline Logit KD (Pha 1)** với **Diagnosis-Driven DPO (Pha 3)** dùng LLM-as-a-Judge. SLM 1.7B chưng cất đạt suy luận in-distribution mạnh nhất trên **NLI (0.8156)** và **Syllogism (0.5684)**; **QLoRA hoạt động như bộ điều chuẩn ngầm** chống quên ở 0.6B, dù các pha alignment nâng cao làm bất ổn quy mô dưới tỷ này. SFT tổng quát hóa OOD tốt hơn, nhưng khung chứng minh tính khả thi của trợ lý suy luận pháp lý nhỏ gọn, an toàn, chạy tại chỗ.

---

# CHƯƠNG 1 — MỞ ĐẦU

## 1.1 Lý do chọn đề tài
Triển khai LLM độc quyền (API) trong quy trình pháp lý gặp 3 rào cản then chốt:
- **Quyền riêng tư dữ liệu:** gửi truy vấn chứa dữ liệu cá nhân/doanh nghiệp ra API ngoài vi phạm chủ quyền dữ liệu (Luật An ninh mạng VN).
- **Chi phí vận hành:** phí token định kỳ cho ngữ cảnh luật dài là không bền vững với tòa án/công ty luật nhỏ.
- **Tính khả dụng:** phụ thuộc mạng → trễ và gián đoạn, không chấp nhận được cho xử lý thông lượng cao.

SLM (<3B tham số) chạy on-premise là giải pháp an toàn, rẻ, trễ thấp — nhưng có **"reasoning gap"**: suy luận đa bước yếu, hay bịa điều luật, rơi vào vòng lặp logic. Trong luật, **một lỗi logic có thể vô hiệu cả bản án**. KD chuyển tri thức từ teacher đã căn chỉnh sang student; nhưng chưng cất offline chuẩn vướng **exposure bias** và không xử được điểm mù hiệu chỉnh (calibration) của student. Đề tài đề xuất khung lai: offline logit KD + tinh chỉnh quỹ đạo on-policy + DPO dẫn dắt bằng chẩn đoán, may đo cho 3 tác vụ pháp lý VN (NLI, MCQ, Syllogism).

## 1.2 Mục tiêu
**Mục tiêu tổng quát:** xây khung suy luận pháp lý **an toàn, hiệu quả tính toán, vững logic**, chưng cất năng lực suy luận từ teacher lớn đã căn chỉnh RL vào một student SLM nhẹ, triển khai được trên phần cứng tại chỗ, giữ độ chính xác/nhất quán khi áp dụng luật VN.

**5 mục tiêu cụ thể:**
- **O1 — Thiết kế khung chưng cất & alignment đa pha:** Offline Logit KD + On-Policy KD + Diagnosis-Driven DPO, tổ chức thành **hai nhánh** từ base SFT chung.
- **O2 — Giảm exposure bias:** on-policy KD huấn luyện trên rollout tự sinh; teacher ở chế độ **Oracle** (được xem đáp án vàng), tích hợp **KL clipping** chặn ảnh hưởng gradient của token định dạng.
- **O3 — Khung alignment dẫn dắt bằng chẩn đoán:** pipeline kiểm định ngữ nghĩa dùng **LLM-as-a-Judge** để chẩn đoán lỗi và tổng hợp cặp preference có mục tiêu, thay cho difficulty-score dựa logits (vốn lệch calibration ở mô hình <3B).
- **O4 — Thích ứng & đánh giá trên benchmark VN:** fine-tune và đánh giá SLM (0.6B, 1.7B) trên NLI, MCQ, Syllogism.
- **O5 — Ablation & phân tích thực nghiệm:** so cấu hình HP, LoRA vs QLoRA, ảnh hưởng quy mô, kèm phân tích định tính.

## 1.3 Đối tượng & Phạm vi
**Đối tượng:**
- **Mô hình:** student auto-regressive <3B — **Qwen3-0.6B-Base** và **Qwen3-1.7B-Base**; teacher **Qwen3-4B-legal-GRPO** (đã căn chỉnh GRPO trên luật VN, **kế thừa từ pha trước — huấn luyện teacher NGOÀI phạm vi**).
- **Dữ liệu:** 3 tác vụ VN — **NLI** (relevant/not_relevant/unknown), **MCQ** (A–D), **Syllogism** (đại tiền đề → tiểu tiền đề → kết luận).

**Phạm vi:**
- **Phương pháp:** chỉ hậu-huấn luyện (post-pretraining); KD (offline + on-policy) và DPO; PEFT là **LoRA/QLoRA**; **không** pre-training nền.
- **Ngôn ngữ/lĩnh vực:** chỉ khung pháp luật & thuật ngữ **tiếng Việt**.
- **Phần cứng:** phản ánh thực tế cơ quan nhỏ — GPU đơn 16GB VRAM (**NVIDIA T4**).

## 1.4 Đóng góp
1. **Pipeline chưng cất & alignment 3 pha** (2 nhánh từ SFT chung) → SLM 1.7B đạt Overall 0.5280 (in-dist).
2. **Bằng chứng thực nghiệm QLoRA chống catastrophic forgetting** ở 0.6B (CE ~0.37 dưới QLoRA vs ~0.56 dưới LoRA; +~33% Syllogism).
3. **Giao thức dựng preference dẫn dắt bằng chẩn đoán** (LLM-as-a-Judge): OK 11.1%→20.7%, phát hiện lỗi 40.5%→63.5%.
4. **Phân tích phân tầng theo quy mô:** on-policy/DPO chỉ lợi **trên ngưỡng dung lượng tối thiểu**; báo cáo trung thực đánh đổi in-dist vs OOD.
5. **Đánh giá tái lập được** trên 3 tác vụ VN, cả in-distribution (Split-Test) và OOD (VLSP2025 Public-Test).

## 1.5 Cấu trúc báo cáo
Ch2 nền lý thuyết · Ch3 phương pháp (formulation + 3 pha) · Ch4 thí nghiệm (dữ liệu, setup, kết quả, ablation) · Ch5 kết luận, giới hạn, hướng phát triển.

---

# CHƯƠNG 2 — CƠ SỞ LÝ THUYẾT

## 2.1 SLM trong lĩnh vực pháp lý
- **Cơ hội/ràng buộc:** SLM (<3B) cho riêng tư tuyệt đối + trễ thấp khi chạy tại chỗ; nhưng dung lượng hạn chế → độ sâu suy luận thấp, dễ ảo giác.
- **Survey khung thích ứng:** **LegalDrill** (tổng hợp dữ liệu dẫn dắt bằng chẩn đoán, chấm khó bằng logits-difficulty); **Italiani et al.** (LLM sinh dữ liệu + KD token-level; soft logits chuyển tri thức tốt hơn hard-label); benchmark VN **LegalSLM Shared Task (VLSP)** và **VLegal-Bench** cho thấy SLM tổng quát lỗi cao khi phân tích cấu trúc điều luật.
- **3 khoảng trống:** (1) **Exposure bias** trong suy luận tuần tự (KD chuẩn dùng teacher-forcing off-policy); (2) **Miscalibration** — difficulty-score dựa logits không tin cậy ở <3B (gán tự tin cao cho chuỗi sai); (3) **Đặc thù pháp hệ VN** — cấu trúc điều-khoản lồng nhau, chưa được tối ưu.

## 2.2 Chưng cất tri thức (KD)
- **KD off-policy cổ điển (Hinton):** student khớp soft target của teacher, làm mượt bằng nhiệt độ T; `L_KD = T²·D_KL(softmax(z_T/T) ‖ softmax(z_S/T))`. Soft target mang **"dark knowledge"** (tương quan giữa các token mà hard label bỏ lỡ). DistilBERT là ví dụ. Nhược: chỉ giám sát trên quỹ đạo do policy ngoài sinh → lệch khi inference.
- **Exposure bias:** train chỉ thấy tiền tố vàng (teacher-forcing), inference phải nối chuỗi tự sinh; lỗi sớm **tích lũy bậc hai** dọc quỹ đạo → hỏng cả suy luận pháp lý.
- **On-policy KD:** student sinh rollout, teacher chấm trên chính tiền tố tự sinh → vòng phản hồi sửa lỗi, học cách "hồi phục" khỏi lỗi.
- **Top-K sparse vs full-vocab:** full-vocab (32k–150k token) tín hiệu giàu, ổn định nhưng tốn bộ nhớ/băng thông; **Top-K (K=50)** giảm mạnh bộ nhớ nhưng bỏ đuôi phân phối (mất dark knowledge âm) và tăng phương sai gradient — đánh đổi cốt lõi.

## 2.3 Tinh chỉnh hiệu quả tham số (PEFT)
- **LoRA:** giả thuyết cập nhật trọng số có "hạng nội tại" thấp; đóng băng W₀, học `ΔW = (α/r)·B·A` với r ≪ min(d,k); B init 0. Giảm ~10⁴ lần tham số huấn luyện; merge được lúc suy luận (không thêm trễ).
- **QLoRA như điều chuẩn ngầm:** đóng băng base ở 4-bit NF4 + adapter 16-bit → ngoài tiết kiệm bộ nhớ, việc **hạn chế dung lượng thích ứng** ép cập nhật low-rank chỉ bắt các mẫu suy luận cốt lõi (thay vì học vẹt tương quan bề mặt) → chống overfitting/quên trên dữ liệu luật nhỏ, chuyên biệt.

## 2.4 Học tăng cường cho suy luận pháp lý
- **GRPO (critic-free):** thay critic bằng so sánh tương đối trong nhóm G rollout; advantage `A_i = (r_i − mean(r))/std(r)`; mục tiêu clip theo tỉ số policy + phạt KL với ref. Học suy luận có cấu trúc qua **reward kiểm chứng được** (khớp tag `<think>`, `<major_premise>`, `<minor_premise>`, `<conclusion>`). *(Đây là cách teacher được tạo — bối cảnh.)*
- **Thông tin đặc quyền (LUPI) → teacher Oracle:** teacher được cấp đáp án vàng `x*` (ẩn với student, vắng lúc inference); tính `P_T(·|q, o, x*)` để chấm rollout với độ tự tin gần tuyệt đối; student tối thiểu `L_oracle = D_KL(P_T(·|q,o,x*) ‖ P_S(·|q,o))` → thừa hưởng logic của oracle mà vẫn chạy ở điều kiện thường.

## 2.5 Tối ưu hóa sở thích & Alignment
- **DPO:** bỏ reward model; dưới Bradley-Terry, `P(y_w≻y_l|x)=σ(r(x,y_w)−r(x,y_l))`, biểu diễn reward ngầm `r(x,y)=β·log(π_θ(y|x)/π_ref(y|x))` → mục tiêu `L_DPO = −E log σ(β·log(π_θ(y_w)/π_ref(y_w)) − β·log(π_θ(y_l)/π_ref(y_l)))`. Trong luật: chosen = giải thích đúng/hợp cấu trúc, rejected = rollout student có lỗi logic.
- **LLM-as-a-Judge:** BLEU/ROUGE (khớp n-gram) không đo được đúng-sai ngữ nghĩa (một từ "không" đảo cả kết luận). Judge (GPT-4-class) chấm theo rubric đa chiều (groundedness, coherence, tag) và **phân loại chẩn đoán** rollout (correct/partial/risky/incorrect) → lọc cặp preference chất lượng cao tự động.

## 2.6 Metric đánh giá trong AI pháp lý
- **Phân loại (NLI, MCQ):** Accuracy, Precision, Recall, F1; **Macro-F1** (trung bình F1 không trọng số qua C lớp) làm chính khi mất cân bằng lớp — bảo đảm lớp hiếm được phản ánh.
- **Sinh (Syllogism):** BLEU (precision n-gram + brevity penalty), **ROUGE-L** (dựa LCS). **Hạn chế cốt lõi:** chỉ đo trùng lặp từ vựng → không đánh giá đúng-sai ngữ nghĩa/logic; đây chính là lý do dùng ROUGE-L làm **proxy** và cần Judge bổ trợ.

---

# CHƯƠNG 3 — PHƯƠNG PHÁP (đào sâu)

## 3.1 Hình thức hóa & tổng quan
Suy luận pháp lý = **sinh chuỗi có điều kiện** `P(y|x)=∏_t P(y_t|y_<t, x)`, với `x=(c,q)` (ngữ cảnh luật + truy vấn) và `y=(s,v)`: `s` = chuỗi suy luận (CoT), `v` = kết luận/nhãn. Mục tiêu: tối ưu `θ` của student `π_θ` xấp xỉ teacher `π_T`, giữ nhẹ và chính xác.

**Kiến trúc 2 nhánh (KHÔNG tuyến tính):** base SFT chung →
- **Nhánh offline:** Pha 1 (trên SFT) → Pha 3 (DPO **trên checkpoint Pha 1**; checkpoint Pha 1 cũng là **reference policy** của DPO).
- **Nhánh on-policy:** Pha 2 áp **thẳng lên SFT**, độc lập Pha 1.
Hai nhánh là **hai chiến lược tinh chỉnh thay thế** được đánh giá song song.

## 3.2 Pha 1 — Offline Logit Distillation
- **Neo hard-label:** CE chỉ trên token đáp án (prompt bị mask −100): `L_CE = −(1/N_out)·Σ log P_student(y_t|y_<t, x)` → giữ student bám đáp án vàng, không chạy theo nhiễu teacher.
- **Soft-label Top-K + nhiệt độ:** chỉ lưu **Top-K=50** logit teacher; chuẩn hóa cả hai phân phối trên V_top với `T>1`; `L_KD = T²·(1/N_out)·Σ_t Σ_{v∈V_top} P^T_teacher(v)·log(P^T_teacher(v)/P^T_student(v))`.
- **Loss tổng:** `L_Phase1 = α₁·L_KD + (1−α₁)·L_CE`, **α₁ = 0.3** (70% CE / 30% KD).
- **QLoRA:** base 4-bit NF4 đóng băng, chỉ cập nhật adapter (Q,K,V,proj) → điều chuẩn ngầm, chống quên thảm họa, ổn định gradient.
- **HP & lý do:** LR **2e-5** (nhỏ, tránh phá trọng số pre-train), α₁=0.3 (neo CE mạnh), **batch hiệu dụng 64** (giảm phương sai gradient → hội tụ mượt). Top-K=50.

## 3.3 Pha 2 — On-Policy Knowledge Distillation
- **Động lực:** vá exposure bias — offline `L_offline` điều kiện trên tiền tố vàng `y*_<t`; inference lại điều kiện trên `ŷ_<t` tự sinh → lỗi tích lũy bậc hai.
- **Rollout tự sinh:** `ŷ_t ~ π_θ(·|x, ŷ_<t)`, decoding ngẫu nhiên **T=0.7, top-p=0.9** để khám phá đa dạng lỗi.
- **Teacher Oracle:** `P_teacher(v|x, y*, ŷ_<t)` — điều kiện trên **đáp án vàng y*** → dù tiền tố student sai, teacher vẫn cho phân phối next-token đúng, tự tin, kéo student về đường đúng.
  - **Rò rỉ & giảm thiểu (nêu thẳng):** cấp `y*` cho teacher là **đường rò rỉ tiềm tàng**; hạn chế bằng (1) tín hiệu KD áp trên **quỹ đạo tự sinh** (không phải chuỗi vàng), (2) **KL clipping** chặn đóng góp token tự tin cao (kể cả token đáp án). Coi rò rỉ còn dư là **vấn đề hiệu lực (validity)** → đưa vào Giới hạn.
- **KL clipping:** token cấu trúc (`<think>`, `<major_premise>`…) entropy cực thấp → KL bùng nổ, lấn át gradient nội dung. Cắt `KL_t^clipped = min(KL_t, C)`, **C = 5.0**, T=1.0.
- **Loss 2 phần:** `L_Phase2 = α₂·L_KD-onpolicy + (1−α₂)·L_CE-anchor`, **α₂ = 0.5**; `L_KD-onpolicy = T²·(1/N_response)·Σ KL_t^clipped`; teacher logits **detach** (mục tiêu hằng). Neo SFT chống quên. **~8× chậm hơn** offline.

## 3.4 Pha 3 — Diagnosis-Driven DPO
- **Pipeline dựng dữ liệu:** student sinh rollout → **LLM Judge (GPT-4o-mini)** kiểm định ngữ nghĩa cả `<think>` lẫn kết luận → nhãn **OK / RISKY / PARTIAL / WRONG**; chỉ **RISKY & WRONG** thành cặp: **chosen = đáp án vàng, rejected = chính lỗi student** → **lấy mẫu phân tầng** cân bằng tác vụ.
- **DPO:** `L_DPO` (mục 2.5), **β = 0.1**, LR **5e-6**, optimizer `adamw_8bit`, 2 epoch, max prompt/response 2048/1024. **Khởi tạo & reference = checkpoint Pha 1.** Trực giác: kéo xác suất câu đúng lên, đẩy câu sai xuống, β giữ không trôi xa ref.

---

# CHƯƠNG 4 — THÍ NGHIỆM & ĐÁNH GIÁ (đào sâu)

## 4.1 Dữ liệu
- **ViLawQA (thangvip):** 3 nguồn HF — MCQ (1,007), NLI (1,421), Syllo (1,000) → gộp **3,428** bản ghi, chuẩn hóa 1 schema; chia **phân tầng theo `task_type`** 80/10/10 → sau làm sạch: **2,603 train · 322 val · 325 test (Split-Test, in-dist)**. ~**75%** mẫu train có chuỗi `<think>` (25% đáp án trực tiếp).
- **Split train theo tác vụ:** MCQ 796 · NLI 1,128 · Syllo 679.
- **OOD:** **VLSP2025 Public-Test 440** (146 MCQ / 150 NLI / 144 Syllo), chuyển về định dạng ViLawQA.
- **Dữ liệu DPO:** bộ ba `(x, y_w, y_l)`. **Rule-based** trích 2,313 cặp (0.6B) nhưng nhiễu; **LLM Judge** cho tập sạch hơn: **2,065 cặp (0.6B)**, **1,434 cặp (1.7B)**.

## 4.2 Thiết lập
- **Teacher:** Qwen3-4B-legal-GRPO, nạp **4-bit NF4** (đóng băng) khi chưng cất.
- **Student init từ SFT checkpoint:** 0.6B (HoangVuSnape/qwen3-0.6b-sft hoặc qlora-sft, huấn luyện **QLoRA**); 1.7B (HoangVuSnape/qwen3-1.7b-sft, **LoRA**).
- **HP theo pha:** P1 (LR 2e-5, α 0.3, T 1.5, Top-K 50, batch 64); P2 (LR 2e-5, α 0.5, T 0.7, top-p 0.9, KL clip 5.0, batch 32); P3 (LR 5e-6, β 0.1, adamw_8bit, 2 epoch).
- **Phần cứng:** **Kaggle 2×T4 16GB** (P1, P3, và P2 của 0.6B — dùng **Model-Split**: student cuda:0, teacher 4-bit cuda:1); **Docker 1×RTX 6000 Ada 48GB** (P2 của 1.7B, batch 8, giảm 10h→5h). Framework: **Unsloth, TRL (`DPOTrainer`), bitsandbytes**.

## 4.3 Kết quả & phân tích
> **Overall** = trung bình **không trọng số** của MCQ-Acc, NLI-Acc, Syllo-ROUGE-L → chỉ báo **thô** (trộn accuracy với lexical); **per-task là cơ sở chính**. Giải mã greedy.

**(a) HP tuning V1→V2:** V1 (LR 1e-4, batch 16, α 0.5) gây **quên thảm họa** (CE 0.3→0.7, sinh chữ trộn Việt–Anh–Trung). V2 (LR 2e-5, batch 64, α 0.3) ổn định (CE 0.5→0.25). 1.7B Overall **0.3644 (E3)→0.5322 (A3)**.

**(b) LoRA vs QLoRA (0.6B, cùng V2):** LoRA (A1) đỉnh sớm ~step 60 rồi quên, CE ~0.56 & trôi lên; QLoRA (A2) CE giảm đều ~0.37, cải thiện tới step 120. **Syllogism R-L 0.1741→0.3804 (~+33%)**, Overall 0.4520→0.4622 → chọn QLoRA cho 0.6B.

**(c) Quy mô 0.6B vs 1.7B:** Judge (GPT-4o-mini) trên train → tỉ lệ **OK 44.9% (1.7B) vs 20.7% (0.6B)**; Syllogism OK **6%→20%**. 1.7B áp đảo 0.6B ở mọi pha (~1.5–2×).

**(d) SFT vs Offline KD (bổ khuyết theo tác vụ):** 0.6B SFT giỏi NLI/Syllo nhưng MCQ chỉ 0.14 → KD nâng MCQ lên **0.4158**, Overall 0.4282→0.4622. 1.7B SFT giỏi MCQ (0.50) yếu Syllo (0.207) → KD nâng Syllo lên **0.4208**, Overall 0.4840→**0.5322**.

**Bảng — tiến trình In-distribution (Split-Test 325), Overall & per-task:**

| Run | Stage | Size | MC | NLI | Syl R-L | Overall |
|---|---|---|---|---|---|---|
| B2 | SFT | 0.6B | 0.140 | 0.716 | 0.428 | 0.4282 |
| A2 | Offline KD | 0.6B | 0.416 | 0.589 | 0.380 | 0.4622 |
| A2 | On-Policy | 0.6B | 0.160 | 0.496 | 0.275 | 0.3104 ▽ |
| A2 | DPO | 0.6B | 0.100 | 0.319 | 0.139 | 0.1861 ▽▽ |
| B3 | SFT | 1.7B | **0.500** | 0.745 | 0.207 | 0.4840 |
| A3 | Offline KD | 1.7B | 0.446 | 0.731 | 0.421 | **0.5322** |
| A3 | On-Policy | 1.7B | 0.360 | 0.787 | 0.375 | 0.5075 |
| A3 | DPO | 1.7B | 0.200 | **0.8156** | **0.5684** | 0.5280 |

**(e) Offline vs On-Policy:** 1.7B on-policy nâng **NLI 0.7305→0.7870**; Overall giảm nhẹ (0.5322→0.5075) do loss on-policy phạt token suy luận tự do → giảm Syllo R-L. **0.6B phản tác dụng** (Overall →0.3104): rollout tự sinh nhiễu vượt sức chứa.

**(f) DPO:** 1.7B — DPO **định hình lại profile**: NLI 0.7305→**0.8156**, Syllo 0.4208→**0.5684**, nhưng MCQ 0.4455→0.20 → Overall 0.5280 (hơi dưới offline 0.5322). **0.6B sụp về 0.1861** (hình phạt probability-displacement quá gắt). ⇒ DPO chỉ lợi ở 1.7B.

**(g) Tổng quát hóa OOD (VLSP2025 440):**

| Run | Stage | Size | MC | NLI | Syl R-L | Overall |
|---|---|---|---|---|---|---|
| B3 | SFT | 1.7B | **0.740** | 0.873 | 0.305 | **0.6395** |
| A3 | On-Policy | 1.7B | 0.555 | **0.920** | **0.372** | 0.6155 |
| A3 | Offline KD | 1.7B | 0.219 | 0.720 | 0.353 | 0.4307 |
| A3 | DPO | 1.7B | 0.212 | 0.893 | 0.282 | 0.4627 |

→ **SFT 1.7B tổng quát hóa tốt nhất (0.6395)** — chưng cất **chuyên biệt hóa** về phân phối train, rõ nhất ở **MCQ** (mọi bản distilled <0.56 vs 0.74). Trong nhóm distilled, **On-Policy 1.7B mạnh nhất (0.6155)**, NLI 0.920 & Syllo 0.372 cao nhất bảng → phá exposure bias giúp tổng quát hóa. **Không có một mô hình tối ưu duy nhất** — phụ thuộc chế độ đánh giá & tác vụ.

## 4.4 Ablation
- **QLoRA vs LoRA (động lực học, 0.6B):** CE loss **LoRA 1.00→0.56 (cao, trôi lên)** vs **QLoRA 0.64→0.37 (giảm đều)** → QLoRA neo tri thức SFT; Syllo R-L 0.1741 vs **0.3804**.
- **Rule-based vs LLM Judge (2,603 rollout 0.6B):**

| Diagnoser | OK | RISKY (đoán may) | PARTIAL | WRONG |
|---|---|---|---|---|
| Rule (Regex+ROUGE-L) | 290 (11.1%) | 898 (34.5%) | 361 (13.9%) | 1,054 (40.5%) |
| LLM Judge (GPT-4o-mini) | **538 (20.7%)** | 404 (15.5%) | 9 (0.3%) | **1,652 (63.5%)** |

→ Judge phục hồi câu đúng-khác-định-dạng (OK↑), hạ "đoán may" (RISKY↓), phát hiện WRONG↑ → tập preference sạch hơn → DPO ổn định hơn.

*(4.5 Qualitative Analysis đã được gỡ khỏi bản báo cáo hiện tại.)*

---

# CHƯƠNG 5 — KẾT LUẬN

## 5.1 Kết luận (5 phát hiện)
1. **Khung đa pha được hiện thực & kiểm chứng:** 2 nhánh từ SFT; 1.7B sau DPO đạt NLI 0.8156 & Syllo 0.5684 (Overall 0.5280, hơi dưới offline 0.5322 — đánh đổi profile).
2. **Lượng tử hóa thiết yếu ở quy mô nhỏ:** QLoRA điều chuẩn ngầm (CE 0.64→0.37 < LoRA ~0.56), +~33% Syllo.
3. **On-policy giảm exposure bias:** NLI 1.7B 0.7305→0.7870 in-dist; OOD cao nhất NLI 0.920 & Syllo 0.372.
4. **Chẩn đoán bằng LLM cho dữ liệu sạch hơn:** OK 11.1→20.7%, WRONG 40.5→63.5%.
5. **Quy mô là yếu tố quyết định:** 1.7B > 0.6B ~1.5–2×; on-policy/DPO chỉ lợi trên ngưỡng dung lượng tối thiểu.

## 5.2 Giới hạn (8)
(1) **Bất ổn dung lượng 0.6B** (on-policy 0.3104, DPO 0.1861). (2) **Chuyên biệt hóa OOD** (SFT 0.6395 cao nhất; MCQ mọi bản distilled <0.56). (3) **Metric proxy & composite thô** (Syllo dùng ROUGE-L, không phải judge chính thức; Overall trộn thang đo). (4) **Chạy một lần**, n=325, không nhiều seed → chênh nhỏ (0.5322 vs 0.5280) có thể trong nhiễu. (5) **Phạm vi hẹp** (1 corpus 2,603 mẫu, 1 lĩnh vực luật VN, 1 teacher). (6) **Rò rỉ đáp án từ teacher Oracle** ở Pha 2 — chưa ablation tách phần học thật vs rò rỉ. (7) **Phụ thuộc API ngoài** (GPT-4o-mini) → cam kết riêng tư chỉ cho **inference/triển khai**, không cho huấn luyện; hạn chế tái lập. (8) **Phần cứng giới hạn** → chưa thử student >1.7B; on-policy ~8× chậm.

## 5.3 Hướng phát triển (4)
1. **Chống quên:** continual learning — replay dữ liệu SFT, điều chuẩn tham số (EWC/L2-SP/KL anchor về SFT), adapter merging.
2. **Alignment nhạy dung lượng cho mô hình nhỏ:** curriculum tăng dần độ khó, hoặc thay DPO bằng **KTO/ORPO/IPO** nhẹ hơn để cứu 0.6B.
3. **Chống chuyên biệt hóa quá mức:** trộn dữ liệu tổng quát + replay buffer, phạt lệch khỏi policy SFT rộng.
4. **Mở rộng student & đa dạng teacher:** chưng cất vào ≥4B, **multi-teacher/self-distillation**, mở rộng dữ liệu & phân ngành luật để nâng trần hiệu năng.
