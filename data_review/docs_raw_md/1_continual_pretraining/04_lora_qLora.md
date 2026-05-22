## CPT với LoRA / QLoRA — cần biết gì

### LoRA vs Full Fine-tuning cho CPT

| | Full FT | LoRA | QLoRA |
|---|---|---|---|
| VRAM | Rất cao | Trung bình | Thấp |
| Học domain knowledge | Tốt nhất | Vừa | Vừa, kém hơn LoRA |
| Catastrophic forgetting | Cao | Thấp | Thấp |
| Tốc độ | Chậm | Nhanh | Nhanh |

**Vấn đề cốt lõi:** LoRA/QLoRA chỉ train **adapter** (ma trận thấp hạng) chèn vào các layer — không sửa trực tiếp weights gốc. Điều này tốt cho SFT, nhưng với CPT thì **knowledge không thấm sâu vào model** bằng full fine-tuning.

---

### Các tham số quan trọng khi CPT với LoRA

**`r` (rank)** — kích thước adapter
- SFT thường dùng `r=8` hoặc `r=16`
- CPT nên dùng **`r=64` hoặc `r=128`** để có đủ capacity học domain knowledge

**`target_modules`** — apply LoRA vào layer nào
- CPT nên target **nhiều layer hơn SFT**: thêm cả `embed_tokens`, `lm_head`, MLP layers
- SFT thường chỉ cần attention layers (`q_proj`, `v_proj`)

**`lora_alpha`**
- Thường set = `r` hoặc `2 * r`

**Learning rate**
- CPT dùng LR **thấp hơn SFT**: `1e-4` đến `5e-5`
- Tránh overfit vào 2 shards nhỏ của bạn

**`num_train_epochs`**
- Dataset nhỏ (~46M tokens) → có thể train **2–3 epochs**

---

### QLoRA thêm điều cần biết

QLoRA = LoRA + base model quantized 4-bit (NF4). Cần lưu ý:

- `bnb_4bit_compute_dtype = bfloat16` — tính toán ở bf16, lưu ở 4-bit
- `bnb_4bit_use_double_quant = True` — double quantization, tiết kiệm thêm VRAM
- Gradient checkpointing bắt buộc bật để fit VRAM

---

## GGUF Quantization — Q1 đến Q8

GGUF là format của **llama.cpp** để chạy model trên CPU/GPU consumer. Quantization giảm bits/weight:

| Level | Bits/weight | Size (7B model) | Chất lượng | Dùng khi |
|---|---|---|---|---|
| **Q2_K** | ~2.6 bit | ~2.8 GB | Rất kém | Chỉ khi RAM cực hạn |
| **Q3_K_M** | ~3.4 bit | ~3.5 GB | Kém | Thử nghiệm nhanh |
| **Q4_0** | 4 bit | ~4.1 GB | Trung bình | CPU phổ thông |
| **Q4_K_M** | ~4.5 bit | ~4.8 GB | **Tốt** | ✅ Cân bằng tốt nhất |
| **Q5_K_M** | ~5.7 bit | ~5.7 GB | Rất tốt | GPU mid-range |
| **Q6_K** | ~6.6 bit | ~6.6 GB | Gần FP16 | GPU có VRAM tốt |
| **Q8_0** | 8 bit | ~8.5 GB | Gần như lossless | Khi có đủ RAM |
| **F16** | 16 bit | ~14 GB | Gốc | Full precision |

**Hậu tố `_K` nghĩa là gì:**
- `_K` = dùng k-quant (quantize theo group, chính xác hơn)
- `_K_S` / `_K_M` / `_K_L` = Small / Medium / Large — cách phân bổ bits giữa các layer

---

### Khuyến nghị thực tế

```
Deploy production  →  Q4_K_M  (cân bằng tốc độ / chất lượng)
Đánh giá nhanh    →  Q5_K_M  (gần FP16 hơn)
RAM rất hạn chế   →  Q3_K_M  (chấp nhận mất chất lượng)
```

Với pipeline của bạn: train CPT bằng QLoRA → merge adapter → export GGUF Q4_K_M để test local bằng llama.cpp hoặc Ollama.


---

# More
Để hiểu đơn giản về các ký hiệu **GGUF (Q2, Q3, Q4...)**, bạn hãy tưởng tượng việc nén mô hình AI cũng giống như việc bạn nén một file video từ độ phân giải gốc 4K xuống Full HD hoặc 720p vậy. Video dung lượng càng nhỏ thì máy yếu càng dễ xem, nhưng hình ảnh sẽ bớt sắc nét đi.

Chữ **Q** là viết tắt của **Quantization** (Lượng tử hóa). Con số đi sau chữ Q chính là **số bit (bit-width)** được dùng để lưu trữ một "trọng số" (weight) của mô hình. Mô hình gốc khi chưa nén thường ở định dạng **FP16 (16-bit)**. Khi chuyển sang GGUF, người ta ép số bit này xuống thấp hơn để giảm dung lượng file và tiết kiệm RAM/VRAM.

Thông thường, GGUF sẽ có các mức nén phổ biến từ **Q2 đến Q8**. Bản **Q1** nguyên bản không có, nhưng hiện tại đã được nâng cấp thành hệ sinh thái **IQ (i-quants/Importance Quants)** cho phép nén xuống tận 1-bit đến 1.5-bit.

---

### Ý nghĩa chi tiết của từng mức Quantization (Q)

| Mức Quantize | Số bit thực tế | Dung lượng file (So với gốc FP16) | Chất lượng giữ lại | Đánh giá & Trải nghiệm thực tế |
| --- | --- | --- | --- | --- |
| **IQ1 / IQ2** | ~1 - 2 bit | Giảm ~85 - 90% | Thấp (~70 - 80%) | **Mức sinh tồn:** Model bị sụt giảm trí thông minh nghiêm trọng, dễ nói nhảm hoặc sai định dạng cấu trúc, chỉ dùng khi máy quá yếu. |
| **Q2_K** | 2-bit | Giảm ~80 - 85% | ~80 - 85% | **Mức khẩn cấp:** Model giữ được một chút kiến thức nền nhưng mất đi khả năng suy luận logic phức tạp. |
| **Q3_K_M** | 3-bit | Giảm ~75% | ~90% | **Mức tiết kiệm:** Tạm ổn cho các bài toán chat thông thường, nhưng với bài toán khó (như Đọc hiểu văn bản Luật) thì bắt đầu "đuối". |
| **Q4_K_M** | 4-bit | Giảm ~70% | **~95 - 97%** | **"Vua quốc dân" (Sweet Spot):** Dung lượng giảm đi gần 4 lần nhưng giữ được hầu như vẹn nguyên bộ não của bản gốc. Đây là bản mặc định khuyên dùng cho mọi người. |
| **Q5_K_M** | 5-bit | Giảm ~60% | ~97 - 98% | **Mức cận cao cấp:** Phục hồi lại các khả năng suy luận sâu bị mất ở bản Q4, cực kỳ thích hợp cho các bài toán cần logic cao. |
| **Q6_K** | 6-bit | Giảm ~55% | ~99% | **Mức cao cấp:** Bắt đầu không thể phân biệt được chất lượng so với bản gốc bằng mắt thường, nhưng file nặng hơn Q4 kha khá. |
| **Q8_0** | 8-bit | Giảm ~50% | ~99.5%+ | **Mức hoàn hảo:** Chất lượng y hệt bản gốc FP16 nhưng dung lượng file nhẹ được một nửa. Có bao nhiêu RAM gánh bấy nhiêu. |

---

### Các ký hiệu đuôi `_S`, `_M`, `_L` hay chữ `_K` nghĩa là gì?

Khi bạn lướt Hugging Face, bạn sẽ thấy những file có tên dài hơn như `Q4_K_M` hoặc `Q3_K_S`. Ý nghĩa của chúng là:

* **Chữ `K` (K-Quants):** Đây là công nghệ nén hỗn hợp hiện đại. Thay vì ép tất cả các block trong mô hình về cùng một số bit, thuật toán sẽ thông minh giữ lại số bit cao ở các tầng quan trọng (ví dụ tầng Attention) và nén mạnh ở các tầng ít quan trọng hơn.
* **Hậu tố `_S` (Small), `_M` (Medium), `_L` (Large):** Thể hiện kích thước file nén trong cùng một mức bit.
* Ví dụ: Cùng là 4-bit nhưng `Q4_K_S` sẽ nén gắt hơn một chút để file nhẹ nhất có thể, còn `Q4_K_M` sẽ nén nương tay hơn ở một số block để giữ chất lượng tốt hơn.



### Tóm lại bạn nên chọn bản nào?

* Nếu bạn ưu tiên **tốc độ nhanh và tiết kiệm RAM** tối đa nhưng mô hình vẫn khôn: Chọn **`Q4_K_M`**.
* Nếu máy bạn dư dả RAM/VRAM một chút và bài toán của bạn cần **độ chính xác, logic cao (như phân tích mã nguồn hoặc văn bản pháp lý)**: Hãy chọn **`Q5_K_M`** hoặc **`Q6_K`** để có trải nghiệm tốt nhất.
