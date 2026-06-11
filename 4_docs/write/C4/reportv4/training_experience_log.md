# Nhật Ký Trải Nghiệm & Bài Học Rút Ra Trong Quá Trình Training

Tài liệu ghi lại toàn bộ trải nghiệm thực tế, lỗi gặp phải, và bài học kinh nghiệm khi huấn luyện pipeline 3-phase distillation (Offline KD → On-Policy KD → DPO) cho mô hình pháp luật Việt Nam.

---

## 1. Tổng Quan Hạ Tầng Đã Sử Dụng

### 1.1 Kaggle — 2×T4 16GB (Miễn phí)

| Thông số | Chi tiết |
|---|---|
| GPU | 2× NVIDIA Tesla T4, mỗi card 16GB VRAM |
| Thời gian tối đa | 12 giờ/session (hard limit) |
| Disk | ~70GB ephemeral |
| Dùng cho | Offline KD (A1, A2, A3), DPO training, On-Policy KD A2 (multi-GPU) |

**Ưu điểm:**
- Miễn phí, đủ cho phần lớn training với model ≤ 1.7B
- Tích hợp sẵn PyTorch, CUDA, HuggingFace
- 2 GPU cho phép model-split strategy (teacher trên GPU 1, student trên GPU 0)

**Hạn chế:**
- 12h time limit → phải ước tính kỹ runtime trước khi chạy
- T4 16GB chật cho On-Policy KD vì cần cả teacher + student + generation buffer cùng lúc
- Mất toàn bộ khi session hết hạn → phải push checkpoint lên HuggingFace Hub thường xuyên
- Không có persistent storage → mỗi lần phải clone repo + install dependencies

### 1.2 Docker — RTX 6000 Ada 48GB (Thuê)

| Thông số | Chi tiết |
|---|---|
| GPU | NVIDIA RTX 6000 Ada Generation, 48GB VRAM |
| Provider | Vast.ai / RunPod (tùy thời điểm) |
| Giá | ~$0.4-0.7/giờ |
| Dùng cho | On-Policy KD A3 (1.7B), Eval nặng, Inference dài |

**Ưu điểm:**
- 48GB VRAM thoải mái: teacher 4bit (~2.5GB) + student 1.7B (~4GB) + optimizer + generation buffer → peak ~25GB, còn dư 23GB
- Không giới hạn thời gian (trả theo giờ)
- Batch size lớn hơn → training nhanh hơn

**Hạn chế:**
- Tốn tiền → phải tối ưu code trước khi chạy, không được debug trên máy thuê
- Setup môi trường mỗi lần: clone repo, install Unsloth + dependencies, set env vars
- Mạng không ổn định → đôi khi push HuggingFace Hub bị timeout

### 1.3 So Sánh Thực Tế

| Tiêu chí | Kaggle 2×T4 | Docker RTX 6000 Ada |
|---|---|---|
| **VRAM khả dụng** | 16GB × 2 (split) | 48GB (single) |
| **A2 On-Policy KD** | bs=4, ~10h ✅ | bs=8, ~5h ✅ |
| **A3 On-Policy KD** | Không đủ VRAM ❌ | bs=8, ~5h ✅ |
| **Chi phí** | $0 | ~$2-4/run |
| **Time limit** | 12h hard | Không giới hạn |

**Bài học:** Dùng Kaggle cho các task vừa (Offline KD, DPO, A2 On-Policy). Thuê GPU cho task cần VRAM lớn (A3 On-Policy 1.7B) hoặc chạy dài. Luôn debug + test trên local/Kaggle trước khi đổ tiền thuê GPU.

---

## 2. Lỗi Training & Cách Khắc Phục

### 2.1 Giai Đoạn Offline KD

#### Bug V1: Catastrophic Forgetting — LR quá cao phá hủy SFT weights

**Triệu chứng:** 
- `ce_loss` tăng dần từ 0.3 → 0.7 qua các epoch (model quên gold labels)
- Eval score đạt peak rất sớm (step 60) rồi sụt giảm liên tục
- Output inference ra rác đa ngôn ngữ (Anh, Trung, Nhật lẫn lộn)

**Nguyên nhân gốc:**
- LR = 1e-4 quá cao cho giai đoạn KD từ checkpoint SFT
- Effective batch size = 16 quá nhỏ, gradient noisy
- alpha = 0.5 (50% KD) → KD loss áp đảo CE loss

**Cách fix (V2):**
```
LR:        1e-4 → 2e-5     (giảm 5x)
alpha:     0.5  → 0.3      (70% CE + 30% KD, bảo vệ reasoning)
eff. batch: 16  → 64       (gradient accumulation tăng)
temp:      2.0  → 1.5      (teacher signal sắc hơn)
eval_steps: 100 → 50       (bắt best checkpoint sớm)
```

**Bài học:** KD từ checkpoint SFT khác hoàn toàn với training từ đầu. LR phải nhỏ hơn SFT 5-10x vì trọng số đã mịn, bước nhảy lớn sẽ phá hủy ngay lập tức.

#### QLoRA là "vũ khí bí mật" chống Forgetting

**Phát hiện:** A2 (QLoRA) tốt hơn A1 (LoRA) ở cùng config, đặc biệt:
- A1: ce_loss tăng (forgetting), eval sụt giảm sau step 60
- A2: ce_loss giảm, eval tăng đều đến step 120

**Lý do:** QLoRA đóng băng base weights ở 4-bit → chỉ train adapter nhỏ → model **không thể ghi đè** kiến thức gốc → hoạt động như regularization tự nhiên.

| Metric | A1 (LoRA) | A2 (QLoRA) |
|---|---|---|
| Best overall | 0.452 (sớm, rồi sụt) | **0.462** (tiếp tục tăng) |
| Syllogism ROUGE-L | 0.285 | **0.380** (+33%) |
| Forgetting | Nặng | Không |

### 2.2 Giai Đoạn DPO

#### Bug: `/think` prefix sai format

**Triệu chứng:** Inference ra rác, model confused
**Nguyên nhân:** Script inference thêm `/think` hoặc `/no_think` vào prompt, nhưng SFT model được train KHÔNG có prefix này
**Fix:** Dùng `build_user_content()` từ `utils.py` giống format training

#### Bug: `do_sample=True` gây diverge

**Triệu chứng:** Model 0.6B sinh text lung tung, không coherent
**Nguyên nhân:** `do_sample=True, temperature=0.6` → model nhỏ quá → sampling gây diverge
**Fix:** Đổi sang `do_sample=False` (greedy decoding) cho inference/eval

#### Bug: NLI accuracy = 0.0

**Triệu chứng:** Tất cả NLI samples đều bị đánh sai
**Nguyên nhân (3 lỗi chồng):**
1. `max_new_tokens=256` quá ngắn → thinking block ăn hết → không còn chỗ cho answer
2. Case sensitivity: extractor check "Không" (hoa) nhưng model output "không" (thường)
3. Repetition loop → model lặp vô hạn

**Fix:**
```python
max_new_tokens: 256 → 1024
answer_extract: thêm .lower()
generation_config: thêm repetition_penalty=1.2
```

#### LLM Judge vs Rule-based: Chất lượng DPO data

**Phát hiện quan trọng:** Rule-based (regex + ROUGE-L) đánh sai rất nhiều:
- 290 samples bị đánh oan là sai → LLM Judge minh oan 538 OK
- ROUGE-L cho qua nhiều answer sai ở Syllogism → LLM Judge phát hiện 1652 WRONG (vs 1054 rule-based)

**Chi phí:** GPT-4o-mini ~$5-7 cho 2603 samples, ~15-30 phút. Rất đáng so với sai lệch data.

### 2.3 Giai Đoạn On-Policy KD — 5 Bugs Nghiêm Trọng

#### Bug 1: Unsloth monkey-patch Qwen3 → teacher 8bit crash

**Triệu chứng:** `AttributeError` khi load teacher ở 8bit sau khi Unsloth đã monkey-patch Qwen3
**Nguyên nhân:** Unsloth thay đổi `RotaryEmbedding` class của Qwen3 khi import → bitsandbytes 8bit không tương thích với class đã bị patch
**Fix:** Chuyển teacher sang **4bit qua Unsloth** (dùng cùng engine):
```python
# TRƯỚC (crash):
teacher = AutoModelForCausalLM.from_pretrained(..., load_in_8bit=True)

# SAU (hoạt động):
teacher, _ = FastLanguageModel.from_pretrained(..., load_in_4bit=True)
```
**Bài học:** Khi dùng Unsloth, TẤT CẢ model (teacher lẫn student) phải load qua Unsloth API, không mix với vanilla transformers. Unsloth monkey-patch ở mức module → ảnh hưởng global.

#### Bug 2: `kd_loss = NaN` do `0 × log(0)` trong KL computation

**Triệu chứng:** KD loss = NaN từ step đầu tiên, toàn bộ training vô nghĩa
**Nguyên nhân:** KL divergence tính thủ công `p * log(p/q)` → khi p=0 hoặc q=0 → NaN
**Fix:** Dùng `F.kl_div(log_target=True)` thay vì tính tay:
```python
# TRƯỚC (NaN):
kl = (teacher_probs * (teacher_probs.log() - student_log_probs)).sum(-1)

# SAU (ổn định):
kl = F.kl_div(
    student_log_probs,
    teacher_log_probs,  # log-space input
    log_target=True,
    reduction='none'
).sum(-1)
```
**Bài học:** Không bao giờ tính KL divergence thủ công cho neural network training. PyTorch `F.kl_div` đã xử lý numerical stability nội bộ.

#### Bug 3: Logit shift lệch 1 position

**Triệu chứng:** KD loss cao bất thường, không giảm dù training
**Nguyên nhân:** Label được shift 1 position (standard causal LM), mask cũng shift, nhưng logits KHÔNG được shift → so sánh lệch:
- Position i của student logits so với position i+1 của teacher logits
- Tức là student đang học predict token TRƯỚC thay vì token ĐÚNG

**Fix:** Shift cả logits trước khi slice:
```python
# TRƯỚC (lệch):
student_logits = student_logits[:, mask_positions, :]

# SAU (đúng):
student_logits = student_logits[:, :-1, :]  # shift logits
# rồi mới apply mask trên shifted logits
```
**Bài học:** Trong causal LM, logits tại position i predict token i+1. Khi so sánh teacher-student logits trên cùng token, cả hai phải được shift đồng nhất.

#### Bug 4: Device mismatch teacher vs student (multi-GPU)

**Triệu chứng:** `RuntimeError: Expected all tensors to be on the same device`
**Nguyên nhân:** Teacher trên cuda:1, student trên cuda:0, tensors không tự chuyển
**Fix:** Thêm `.to(device)` tường minh ở mọi chỗ transfer:
```python
teacher_logits = teacher_logits.to(device)  # cuda:1 → cuda:0
# hoặc ngược lại khi cần
student_ids = student_ids.to(teacher_device)  # cuda:0 → cuda:1
```

#### Bug 5: `kd_stats` undefined khi skip empty generation

**Triệu chứng:** `NameError: name 'kd_stats' is not defined` random giữa training
**Nguyên nhân:** Khi student generate ra chuỗi rỗng → skip KD step → nhưng code sau đó vẫn reference `kd_stats` để log
**Fix:** Init defaults trước vòng if:
```python
kd_stats = {"kd_loss": 0.0, "kl_before": 0.0, "kl_after": 0.0, "num_tokens": 0}
if len(student_output) > 0:
    kd_stats = compute_kd_loss(...)
```

---

## 3. Chiến Lược Multi-GPU trên Kaggle (2×T4)

### 3.1 Tại sao không dùng DataParallel?

Unsloth load model ở 4bit bằng cách monkey-patch layers → các layer đã bị patch **không tương thích** với `nn.DataParallel` hay `DistributedDataParallel`. Cụ thể:
- `DataParallel` clone model sang GPU thứ 2 → clone bị lỗi vì Unsloth custom layers
- `DistributedDataParallel` cần `torch.distributed.init_process_group` → conflict với Unsloth

### 3.2 Giải pháp: Model Split

```
┌──────────────────┐    ┌──────────────────┐
│    GPU 0 (T4)    │    │    GPU 1 (T4)    │
│                  │    │                  │
│  Student (0.6B)  │    │  Teacher (4B)    │
│  - Forward pass  │    │  - Forward pass  │
│  - Backward pass │    │  - 4bit frozen   │
│  - Optimizer     │    │  - torch.no_grad │
│                  │    │                  │
│  ~8-10GB used    │    │  ~6-8GB used     │
└──────────────────┘    └──────────────────┘
         │                      │
         └──── Logits transfer ─┘
              (cuda:1 → cuda:0)
```

**Cách hoạt động:**
1. Input batch trên cuda:0 (student device)
2. Student generate trên cuda:0
3. Copy student output → cuda:1
4. Teacher forward trên cuda:1  
5. Copy teacher logits → cuda:0
6. Compute KL loss + backward trên cuda:0

**File:** `src/train/train_kd_onpolicy_multigpu.py`

### 3.3 Kết quả A2 Multi-GPU

| Metric | Giá trị |
|---|---|
| Batch size | 4 |
| Gradient accumulation | 8 |
| Effective batch | 32 |
| Total steps | 326 (2 epochs) |
| Runtime | ~10h (dưới 12h limit ✅) |
| OOM skips | 5/651 steps (~0.8%) |
| Best overall | **0.4216** |

---

## 4. Xử Lý OOM (Out of Memory)

### 4.1 Try-Catch OOM Skip

Chiến lược chính: wrap toàn bộ training step trong try-catch, skip batch bị OOM:

```python
try:
    loss, stats = training_step(batch)
    loss.backward()
    optimizer.step()
except torch.cuda.OutOfMemoryError:
    print(f"⚠️ OOM at step {step}, skipping batch")
    torch.cuda.empty_cache()
    optimizer.zero_grad()
    oom_count += 1
    continue
```

**Tại sao an toàn:** 
- Mỗi step là independent (không accumulate state giữa các batch)
- Skip 1-2 batch trên hàng trăm steps không ảnh hưởng convergence
- Tốt hơn crash giữa chừng mất hết progress

### 4.2 Chunking — Thử và Bỏ

**Ý tưởng:** Batch size 16, split thành 4 chunks × 4 samples, forward từng chunk rồi accumulate gradient

**Kết quả:** Chậm hơn batch size 8 thuần → overhead từ chunk management + multiple forward passes lớn hơn lợi ích VRAM tiết kiệm

**Kết luận:** Batch size 8 + OOM skip là chiến lược tối ưu cho RTX 6000 48GB.

### 4.3 VRAM Thực Tế

| Setup | Teacher | Student | Optimizer | Buffer | Total | GPU |
|---|---|---|---|---|---|---|
| A2 Kaggle (bs=4) | 2.5GB | 1.5GB | 2GB | 8GB | ~14GB | T4 16GB ✅ |
| A3 Docker (bs=8) | 2.5GB | 4GB | 4GB | 10GB | ~25GB | 6000 48GB ✅ |
| A3 Kaggle (bs=4) | 2.5GB + 4GB = 6.5GB trên 1 GPU | | | | >16GB | T4 ❌ |

---

## 5. Kết Quả Training Tổng Hợp

### 5.1 Offline KD V1 vs V2

| | V1 (config sai) | V2 (config sửa) |
|---|---|---|
| LR | 1e-4 | 2e-5 |
| alpha | 0.5 | 0.3 |
| Eff. batch | 16 | 64 |
| A1 (0.6B LoRA) | Rác, forgetting nặng | Không chạy lại |
| A2 (0.6B QLoRA) | — | **0.462 overall** ✅ |
| A3 (1.7B LoRA) | Forgetting nhẹ | **6/9 test samples** ✅ |

### 5.2 DPO (từ KD checkpoint)

| Model | Hub | DPO Pairs | Data Source |
|---|---|---|---|
| A2 KD+DPO | `qwen3-0.6b-legal-kd-dpo-qlora` | 2065 | LLM Judge (GPT-4o-mini) |
| A3 KD+DPO | `qwen3-1.7b-legal-kd-dpo-lora` | 1434 | LLM Judge (GPT-4o-mini) |

### 5.3 On-Policy KD

**A3 (1.7B, RTX 6000 48GB):**
| Metric | Giá trị |
|---|---|
| Best overall | **0.4415** |
| NLI accuracy | 0.794 |
| MC accuracy | 0.396 |
| Syllogism ROUGE-L | 0.134 |
| Runtime | ~5h |

**A2 (0.6B, Kaggle 2×T4 multi-GPU):**
| Metric | Giá trị |
|---|---|
| Best overall | **0.4216** |
| NLI accuracy | 0.674 |
| MC accuracy | 0.218 |
| Syllogism ROUGE-L | 0.352 |
| Runtime | ~10h |

---

## 6. Bài Học Tổng Hợp

### 6.1 Về Config & Hyperparameter

1. **LR cho KD phải nhỏ hơn SFT 5-10x** — SFT weights đã mịn, bước nhảy lớn phá hủy ngay
2. **alpha (KD weight)**: Offline KD dùng 0.3 (bảo vệ reasoning), On-Policy KD dùng 0.5 (ưu tiên phá Exposure Bias)
3. **Effective batch size ≥ 64** cho Offline KD để gradient ổn định
4. **QLoRA > LoRA cho model nhỏ (≤ 0.6B)** — 4bit frozen base = regularization tự nhiên

### 6.2 Về Infrastructure

5. **Debug local → Test Kaggle → Run GPU thuê** — không bao giờ debug trên máy tính tiền
6. **Push checkpoint lên Hub sau mỗi eval** — Kaggle session có thể chết bất cứ lúc nào
7. **Model split (teacher/student trên 2 GPU) hoạt động tốt** khi DataParallel không khả dụng (Unsloth)
8. **OOM try-catch skip** tốt hơn giảm batch size quá nhỏ — 5 skip / 651 steps = 0.8%, không ảnh hưởng

### 6.3 Về Debugging

9. **Luôn check loss components riêng** (ce_loss, kd_loss) — tổng loss có thể giảm nhưng 1 component tăng = forgetting
10. **NaN loss 99% do numerical instability** — dùng PyTorch built-in functions (F.kl_div) thay vì tính tay
11. **Shift alignment trong causal LM** — logits[i] predict token[i+1], phải shift nhất quán
12. **Test inference trước khi evaluate** — `max_new_tokens`, `do_sample`, `repetition_penalty` đều ảnh hưởng kết quả

### 6.4 Về Methodology

13. **Exposure Bias là vấn đề thực** — Offline KD train trên gold text, student trượt khi tự generate
14. **On-Policy KD giải quyết Exposure Bias** — student generate → teacher chấm → học từ sai lầm của chính mình
15. **Privileged Information (PI) tăng chất lượng teacher scoring** — cho teacher xem gold answer trước khi chấm
16. **KL Clipping ngăn XML token domination** — `<think>`, `<major_premise>` có KL rất lớn, chiếm hết gradient nếu không clip
17. **LLM Judge chính xác hơn Rule-based cho DPO data** — đáng tốn $5-7 để có data sạch

### 6.5 Về Unsloth

18. **Unsloth monkey-patch là global** — sau khi import, MỌI model Qwen3 đều bị patch, kể cả teacher
19. **Không mix Unsloth + vanilla transformers** — load tất cả model qua `FastLanguageModel.from_pretrained()`
20. **4bit qua Unsloth ổn định hơn 8bit qua bitsandbytes** khi dùng chung Unsloth

---

## 7. Timeline Dự Án

```
Tuần 1-2: CPT + SFT (base training)
    ↓
Tuần 3:   Offline KD V1 → phát hiện catastrophic forgetting
    ↓
Tuần 4:   Phân tích nguyên nhân, fix config → KD V2
    ↓
Tuần 5:   KD V2 training (A1, A2, A3) + Eval 9 samples
    ↓
Tuần 6:   DPO pipeline: Student inference → Diagnosis → Build pairs
          So sánh Rule-based vs LLM Judge
    ↓
Tuần 7:   DPO training + Design On-Policy KD
    ↓  
Tuần 8:   On-Policy KD implementation: 5 bugs, multi-GPU script
          A3 trên RTX 6000, A2 trên Kaggle 2×T4
    ↓
Tuần 9:   Eval tổng hợp + Report
```

---

*Cập nhật lần cuối: 2026-06-09*
