# Glossary — Thuật ngữ ML / LLM Fine-tuning

Giải thích các thuật ngữ xuất hiện trong project, wandb dashboard, và config.

---

## Training Metrics

### loss
Sai số giữa output model dự đoán và ground truth. Được tính bằng cross-entropy:
- **Cao** (>2.0): model chưa học được gì
- **Trung bình** (0.8–1.5): đang học
- **Thấp** (<0.6): model học tốt (hoặc overfit nếu eval/loss không giảm theo)

### train/loss
Loss trên **training set** — tính trong lúc đang train. Luôn giảm nếu model có đủ capacity.

### eval/loss
Loss trên **validation set** — tính sau mỗi `eval_steps`. Chỉ số quan trọng nhất để phát hiện **overfit**:
- `eval/loss` giảm cùng `train/loss` → **đang học tốt**
- `eval/loss` tăng trong khi `train/loss` giảm → **overfit** → dừng sớm

### grad_norm (Gradient Norm)
Độ lớn tổng hợp của tất cả gradients trong một bước update:
- **0.1–1.0**: ổn định, bình thường
- **>10**: gradient exploding → cần giảm `learning_rate` hoặc dùng `max_grad_norm`
- **~0**: gradient vanishing → model không học được

### learning_rate (LR)
Bước nhảy khi update weights mỗi iteration:
- **Quá cao** (>1e-3): model dao động, không hội tụ
- **Quá thấp** (<1e-5): học rất chậm
- **Thông thường cho LoRA**: `1e-4` đến `3e-4`

---

## Training Configuration

### epoch
Một lần duyệt qua **toàn bộ** training dataset. `num_train_epochs: 3` = dataset được học 3 lần.

### per_device_train_batch_size
Số samples xử lý **song song** trên 1 GPU mỗi forward pass. Batch lớn hơn:
- Nhanh hơn (GPU utilization cao hơn)
- Tốn VRAM hơn
- Gradient ổn định hơn (trung bình nhiều samples hơn)

### gradient_accumulation_steps
Số mini-batch tích lũy gradient trước khi thực hiện 1 lần update weights:
```
effective_batch_size = per_device_train_batch_size × gradient_accumulation_steps
```
Dùng khi muốn effective batch lớn nhưng VRAM không đủ để fit toàn bộ.

### effective_batch_size
Tổng số samples thật sự ảnh hưởng đến mỗi lần update model:
```
16 (batch) × 2 (grad_accum) = 32 effective batch
```
Batch lớn → training ổn định hơn, cần ít epochs hơn.

### warmup_ratio
Tỉ lệ steps đầu dùng để **tăng dần learning rate** từ 0 → `learning_rate`:
- `warmup_ratio: 0.1` = 10% steps đầu là warmup
- Tránh "shock" model với LR cao ngay từ đầu
- Trên wandb thấy đường LR tăng rồi giảm cosine

### lr_scheduler_type
Cách learning rate thay đổi theo thời gian:
- `cosine`: tăng trong warmup → giảm dần theo đường cong cosine → gần 0 khi kết thúc. **Phổ biến nhất**
- `linear`: giảm thẳng đều
- `constant`: không đổi sau warmup

### weight_decay
Regularization — thêm penalty vào loss tỉ lệ với độ lớn của weights:
- Ngăn model "overconfident" vào một số weights cụ thể
- Giảm overfit
- Thông thường: `0.01` đến `0.1`

### gradient_checkpointing
Kỹ thuật tiết kiệm VRAM bằng cách **không lưu activations** trong forward pass, thay vào đó tính lại khi cần trong backward pass:
- Tiết kiệm ~60% VRAM
- Chậm hơn ~20-30%
- Unsloth dùng variant `"unsloth"` tối ưu hơn nữa

---

## Quantization

### 4-bit Quantization (QLoRA)
Nén weights từ float32 (32-bit) xuống 4-bit:
- 1.7B model: float32 = ~6.8GB → 4bit = ~0.9GB VRAM
- Mất một chút độ chính xác nhưng không đáng kể với LoRA fine-tuning

### NF4 (Normal Float 4)
Kiểu quantization 4-bit được tối ưu cho **phân phối chuẩn (normal distribution)** của weights neural network — chính xác hơn INT4 thông thường.

### double_quant (Double Quantization)
Quantize luôn cả các hằng số quantization → tiết kiệm thêm ~0.4 bit/parameter.

### bfloat16 (bf16)
Định dạng số thực 16-bit với **dải số rộng hơn** float16 (8-bit exponent thay vì 5-bit). Ít bị overflow hơn, phù hợp training LLM trên GPU Ampere+ (RTX 3000+, A100).

---

## LoRA / PEFT

### LoRA (Low-Rank Adaptation)
Thay vì train toàn bộ model, chèn 2 ma trận nhỏ A và B vào mỗi layer:
```
output = W_frozen(x) + (A × B)(x)
W: 2048×2048 = 4M params  →  A: 2048×r + B: r×2048 = 65K params (r=16)
```
Chỉ train A và B → tiết kiệm 98% params cần update.

### r (LoRA rank)
Kích thước "cổ chai" của ma trận LoRA. Rank cao hơn:
- Capacity lớn hơn (học được nhiều hơn)
- Tốn VRAM hơn
- Thông thường: `8`, `16`, `32`, `64`

### alpha (LoRA alpha)
Scaling factor áp dụng lên LoRA output: `scale = alpha / r`
- `alpha = 2×r` (ví dụ r=16, alpha=32) → scale = 2.0 → học nhanh/mạnh hơn
- `alpha = r` → scale = 1.0 → conservative hơn

### target_modules
Các layers được gắn LoRA adapter. Qwen3/Qwen2 architecture:
- **Attention**: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **MLP/FFN**: `gate_proj`, `up_proj`, `down_proj`
- Train nhiều modules hơn = học được nhiều hơn nhưng tốn VRAM hơn

### adapter
File lưu LoRA weights sau khi train (`adapter_model.safetensors` + `adapter_config.json`). Nhỏ hơn full model rất nhiều (~50MB so với ~3.5GB).

---

## Optimizer

### AdamW
Optimizer phổ biến nhất cho LLM. Adam + weight decay decoupled. Lưu 2 moment states (m, v) cho mỗi parameter → tốn gấp 2-3x VRAM so với model.

### adamw_8bit
AdamW nhưng moment states được quantize xuống 8-bit (bitsandbytes). Tiết kiệm ~50% VRAM cho optimizer states mà không ảnh hưởng đáng kể đến kết quả.

### paged_adamw_8bit
`adamw_8bit` + **paging**: khi GPU VRAM đầy, optimizer states được offload sang RAM tự động. Phù hợp GPU nhỏ (<8GB).

---

## Evaluation Metrics

### accuracy
Tỉ lệ dự đoán đúng trên tổng số samples. Dùng cho **NLI** và **Multi-choice**.

### F1 score
Trung bình điều hòa của Precision và Recall:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
Phù hợp khi dataset **imbalanced** (số lượng nhãn không đều).

### ROUGE-1, ROUGE-2, ROUGE-L
Đo độ giống nhau giữa text model sinh ra và ground truth (dùng cho **Syllogism**):
- **ROUGE-1**: overlap từ đơn (unigrams)
- **ROUGE-2**: overlap cụm 2 từ (bigrams)  
- **ROUGE-L**: longest common subsequence — quan trọng nhất, đo cấu trúc câu

---

## Unsloth Specific

### padding-free
Unsloth bỏ padding token, pack sequences liền nhau → không lãng phí compute trên padding → nhanh hơn ~10-20%.

### double buffering
Overlap 2 quá trình: chuyển data từ RAM → GPU (H2D transfer) đồng thời với tính toán forward pass → giảm idle time của GPU.

### gradient offloading
Khi VRAM gần đầy, unsloth tự động offload gradient sang RAM trong backward pass → tránh OOM.

---

## Inference

### thinking mode (Qwen3)
Qwen3 hỗ trợ 2 mode:
- **Thinking ON** (`<think>\n`): model viết reasoning chain trước khi trả lời → chậm hơn nhưng chính xác hơn
- **Thinking OFF** (`<think>\n\n</think>\n\n`): bỏ qua reasoning → nhanh, trả lời trực tiếp

### temperature
Độ "ngẫu nhiên" khi sample token:
- `0`: luôn chọn token có xác suất cao nhất (deterministic)
- `1.0`: sample theo xác suất gốc
- `>1.0`: sáng tạo hơn, có thể sai nhiều hơn
- Eval dùng `temperature=1.0, do_sample=False` = greedy decoding

### greedy decoding
Luôn chọn token có xác suất cao nhất tại mỗi bước. Deterministic, phù hợp evaluation.
