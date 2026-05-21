6GB / 24GB VRAM thì còn dư rất nhiều. Có thể tăng batch lên 32 hoặc 64. Giải thích từng dòng:

---

**Batch & tốc độ:**

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `per_device_train_batch_size` | 16 | Số samples xử lý cùng lúc trên GPU. **Tăng = dùng VRAM nhiều hơn nhưng nhanh hơn** |
| `gradient_accumulation_steps` | 2 | Tích lũy gradient qua 2 mini-batch trước khi update weights |
| **Effective batch size** | 16 x 2 = **32** | Tổng samples mỗi lần update. Batch lớn → training ổn định hơn |

**Epochs:**

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `num_train_epochs` | 3 | Duyệt qua toàn bộ dataset 3 lần. Nhiều epoch → model học kỹ hơn, nhưng quá nhiều → overfit |

**Learning rate & optimizer:**

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `learning_rate` | 2e-4 | Tốc độ học = 0.0002. Quá cao → model không hội tụ, quá thấp → học chậm |
| `optim` | adamw_8bit | Optimizer AdamW dùng 8-bit → tiết kiệm ~50% RAM cho optimizer states |
| `lr_scheduler_type` | cosine | LR bắt đầu cao, giảm dần theo đường cong cosine → kết thúc training êm |
| `warmup_ratio` | 0.1 | 10% steps đầu tăng dần LR từ 0 → 2e-4, tránh shock model lúc đầu |
| `weight_decay` | 0.01 | Regularization — phạt weights lớn, giảm overfit |

**Precision:**

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `bf16` | true | Dùng bfloat16 (16-bit) thay vì float32 → nhanh 2x, tiết kiệm VRAM 2x |
| `fp16` | false | Không dùng float16 (bf16 tốt hơn trên Ampere+) |
| `gradient_checkpointing` | true | Đánh đổi tốc độ lấy VRAM: tính lại activation thay vì lưu → tiết kiệm ~60% VRAM |

**Checkpoint & eval:**

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `eval_strategy` | steps | Eval mỗi N steps (không phải mỗi epoch) |
| `eval_steps` | 200 | Mỗi 200 steps chạy eval trên validation set |
| `save_steps` | 200 | Mỗi 200 steps lưu checkpoint |
| `save_total_limit` | 3 | Giữ tối đa 3 checkpoints gần nhất, xóa cũ → tiết kiệm disk |
| `report_to` | wandb | Gửi metrics lên Weights & Biases để theo dõi realtime |

---

**Với 6GB / 24GB, tăng thêm:**

```yaml
per_device_train_batch_size: 32
gradient_accumulation_steps: 2
# effective batch = 64
```

Nhưng đang train rồi thì **để nó chạy xong** (~20 phút) — lần sau tăng batch thêm.