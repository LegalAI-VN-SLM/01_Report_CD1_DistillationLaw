Dưới đây là giải thích từng bức (đã đối chiếu số liệu thật từ W&B, đúng phạm vi **train/valid dynamics**):

---

## 🟥🟧 Hình 1 — Phase 1: Learning-Rate Tuning & CE-Loss Stability (V1 vs V2)
**Vẽ gì:** CE loss trên gold labels của 2 run **LoRA 0.6B** khác nhau ở learning rate — E1 (V1, LR `1e-4`, đỏ) và A1 (V2, LR `2e-5`, cam). Đường mờ = raw, đường đậm = EMA.

**Data nói gì:**
- E1 (đỏ): CE đi `0.664 → 0.568` (EMA) nhưng **dao động rất mạnh** (raw 0.29–1.13) → học không ổn định, gần như đứng yên quanh ~0.57.
- A1 (cam): từ warmup `1.0 → 0.56`, **giảm mượt hơn**.

**Ý nghĩa:** LR cao (V1) khiến loss nhiễu loạn, không hội tụ rõ; giảm LR (V2) giúp đường loss ổn định. ⚠️ Lưu ý: hai đường **kết thúc gần bằng nhau (~0.56)**, nên CE *không* chứng minh được V2 "giữ kiến thức" tốt hơn — lợi ích thật của V2 nằm ở **điểm test** (Table 4.2: Overall 0.256 → 0.452), không ở chart này.

---

## 🟧🟦 Hình 2 — Ablation: QLoRA vs Standard LoRA (CE trajectory)
**Vẽ gì:** Cùng cấu hình V2 trên 0.6B, so **LoRA A1** (cam, nét đứt, base trainable) với **QLoRA A2** (xanh, base đóng băng 4-bit NF4).

**Data nói gì:**
- LoRA (cam): CE `1.0 → 0.56`.
- QLoRA (xanh): CE `0.64 → 0.37` — **luôn nằm thấp hơn LoRA suốt quá trình**.

**Ý nghĩa:** Đóng băng base ở 4-bit (QLoRA) hoạt động như **bộ điều chuẩn ẩn** → CE thấp hơn, giữ kiến thức SFT tốt hơn. Đây là điểm nhấn **đúng và chắc**. 

---

## 🟩🟦 Hình 3 — Phase 2: Model Size & On-Policy KD Train-Loss Stability
**Vẽ gì:** Total combined loss trong on-policy KD: **0.6B (A2, xanh lá nét đứt)** vs **1.7B (A3, xanh dương)**.

**Data nói gì:**
- 0.6B: hội tụ về **~1.07**, nhiễu nhiều, plateau cao.
- 1.7B: hội tụ mượt về **~0.73**, sau cú warmup spike (~2.5).

**Ý nghĩa:** Mô hình lớn hơn (1.7B) **hấp thụ được tín hiệu on-policy** và hội tụ thấp/ổn định; 0.6B kẹt ở mức cao, nhiễu hơn → kém ổn định. (Từ "collapse" thật sự thuộc về **test**, nên ở đây mình chỉ nói "plateau cao & nhiễu".)

---

## 🟡🩵 Hình 4 — Phase 2: Token-Level KL Stays Below the Clip Threshold
**Vẽ gì:** KL token-level trung bình **trước clip** (vàng) và **sau clip** (xanh cyan), kèm đường ngưỡng **C = 5.0** (đỏ chấm).

**Data nói gì:**
- KL raw cao nhất chỉ **~1.11**, giảm dần về ~0.30.
- Đường before ≈ after (1.11 vs 1.08) → **gần như trùng khít**.
- Ngưỡng 5.0 **không bao giờ bị chạm**.

**Ý nghĩa:** Trong run này, KL trung bình **luôn nằm rất xa dưới ngưỡng** → **clipping hầu như không kích hoạt**, đóng vai trò "rào an toàn" hơn là cơ chế chính. (Đây là lý do mình đã sửa nhãn cũ "chặn format-token dominance" — vì data cho thấy nó không chặn gì đáng kể.)

---

## 🟦⬜ Hình 5 — Phase 3: DPO Train-Set Reward Margin (Chosen − Rejected)
**Vẽ gì:** Reward margin (điểm chosen trừ rejected) trên **train set**: **1.7B (A3, xanh)** vs **0.6B (A2, xám nét đứt)**.

**Data nói gì:**
- 1.7B: `−13.7 → +30` — tách biệt sở thích mạnh, đều.
- 0.6B: `−13.4 → +22` — **cũng nở mạnh**, chỉ thấp hơn 1.7B một chút.

**Ý nghĩa:** Trên **train**, cả hai đều học tách chosen/rejected thành công (margin tăng vọt). ⚠️ Điểm tinh tế: margin train cao của 0.6B **không** đồng nghĩa nó tốt — vì trên **test** nó sụp (Overall 0.1861). Đây chính là dấu hiệu **overfit/reward-hacking**: thắng trên train nhưng hỏng khi tổng quát hóa. (Mình đã gỡ nhãn sai "fail to expand" vì đường train rõ ràng *có* expand.)

---

### 🧭 Mạch chung của 5 hình
| Hình | Pha | Thông điệp train-level |
|---|---|---|
| 1 | P1 | LR thấp (V2) ổn định hơn LR cao (V1) |
| 2 | P1 | QLoRA giữ CE thấp hơn LoRA → điều chuẩn ẩn |
| 3 | P2 | 1.7B hội tụ ổn định; 0.6B nhiễu/cao |
| 4 | P2 | KL không chạm ngưỡng → clip là rào an toàn |
| 5 | P3 | Cả hai tách sở thích trên train; 0.6B hỏng ở test (overfit) |

**Lằn ranh quan trọng:** Hình 1–5 là **động lực huấn luyện** — chúng giải thích *quá trình*. Còn *kết luận thắng/thua thật sự* (collapse, over-specialization) nằm ở **bảng test 4.3/4.4**. Đừng để hội đồng nhầm hai tầng này.

Bạn muốn mình viết các giải thích này thành **caption gọn** đặt dưới mỗi hình trong report (.tex), hay để mình cập nhật luôn **Table 4.5** cho khớp data?

