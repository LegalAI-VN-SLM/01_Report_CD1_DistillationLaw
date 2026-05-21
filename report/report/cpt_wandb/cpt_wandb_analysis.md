# CPT W&B Analysis (LoRA vs QLoRA)

Generated: 2026-05-21 (refined analysis)

## Data sources

- E:\DoCode\1 VN-Legal-AI\legal-slm-finetune\1_download_wandb_evaluate\data_wandb\CPT\qwen3-4b-cpt-lora
- E:\DoCode\1 VN-Legal-AI\legal-slm-finetune\1_download_wandb_evaluate\data_evaluate\CPT_lora_gwen3_4b\eval_results.json
- E:\DoCode\1 VN-Legal-AI\legal-slm-finetune\1_download_wandb_evaluate\data_wandb\CPT\qwen3-4b-cpt-qlora
- E:\DoCode\1 VN-Legal-AI\legal-slm-finetune\1_download_wandb_evaluate\data_evaluate\CPT_qlora_gwen3_4b\eval_results.json

Note: eval_results.json la ket qua danh gia sau khi train (khong phai eval trong qua trinh train).

## Dataset split context

- Total docs: 7958
- Train docs: 7559 (95%)
- Test docs: 399 (5%)

## Run setup (config snapshot)

### Thong so chung

- Model type: qwen3, seq_len=2048, max_position_embeddings=40960
- Epochs: 1, lr=1e-4, scheduler=cosine, warmup_ratio=0.05
- Optim: adamw_8bit, bf16, gradient_checkpointing=true
- Packing: true, padding_free: true
- LoRA: r=64, dropout=0, target_modules=[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

### Khac biet chinh (LoRA vs QLoRA)

| Aspect | LoRA | QLoRA |
| --- | --- | --- |
| Base model | unsloth/Qwen3-4B | unsloth/Qwen3-4B-bnb-4bit |
| LoRA alpha | 64 | 128 |
| Quantization | none | 4-bit NF4 (bnb) |
| per_device_train_batch_size | 4 | 8 |
| gradient_accumulation_steps | 4 | 2 |
| per_device_eval_batch_size | 1 | 2 |

Ghi chu: effective batch size tuong duong (4x4 vs 8x2 = 16).

## Summary table (W&B train + offline eval)

| Run | State | Train loss (last) | Train lr (last) | Grad norm (last) | Global step | Eval loss | Perplexity | Model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LoRA | finished | 0.6338 | 0.00000019 | 0.1738 | 451 | 0.6197 | 1.86 | unsloth/Qwen3-4B |
| QLoRA | failed | 0.6047 | 0.00000011 | 0.2267 | 448 | 0.6034 | 1.83 | unsloth/Qwen3-4B-bnb-4bit |

## Training dynamics (history.csv)

| Run | Loss start -> min -> end | LR start -> end | Avg grad_norm | Rows | Global step end |
| --- | --- | --- | --- | --- | --- |
| LoRA | 1.0367 -> 0.6061 -> 0.6338 | 8.26e-05 -> 1.94e-07 | 0.1857 | 23 | 451 |
| QLoRA | 1.0461 -> 0.5989 -> 0.6047 | 8.26e-05 -> 1.11e-07 | 0.2256 | 23 | 448 |

## Offline eval results

| Run | Eval loss | Perplexity | Total tokens | Adapter | Timestamp |
| --- | --- | --- | --- | --- | --- |
| LoRA | 0.6197 | 1.86 | 745,248 | outputs/qwen3-4b-cpt-lora/adapter_final | 2026-05-16 10:44:03 |
| QLoRA | 0.6034 | 1.83 | 745,248 | outputs/qwen3-4b-cpt-qlora/adapter_final | 2026-05-16 10:46:41 |

## Plots

- ![Train loss](train_loss.png)
- ![Learning rate](learning_rate.png)
- ![Grad norm](grad_norm.png)

## Observations (chi tiet)

- QLoRA tot hon nhe ve train loss (0.6047 vs 0.6338) va eval loss (0.6034 vs 0.6197), chenhlech ~0.029 train loss va ~0.016 eval loss.
- Perplexity QLoRA thap hon ~0.03 (1.83 vs 1.86), chenh lech nho nhung on dinh voi xu huong train loss.
- QLoRA co grad_norm trung binh cao hon (0.2256 vs 0.1857), goi y update co the on ao hon du LoRA.
- Ca hai run chi 1 epoch, loss giam nhanh tu ~1.04 xuong ~0.60; chua co theo doi eval theo thoi gian (eval_strategy=no).
- Thoi gian train va toc do gan tuong duong (train_runtime ~2390s, steps/s ~0.187-0.188), khac biet chu yeu nam o base model va alpha.

## Caveats / can than

- QLoRA run state = failed, ket qua eval chi nen xem la partial; can xem log nguyen nhan fail.
- Config snapshot cho thay do_train=false va eval_strategy=no, co the la artifact cua pipeline export; can doi chieu script chay thuc te.
- Warmup_steps dang la 0.05 (float); can xac nhan framework co hieu day la ty le hay so step.

## Recommendations

- Rerun QLoRA voi checkpoint/monitor de xac nhan on dinh (tranh fail giua chung).
- Them eval_strategy=steps de co duong cong eval trong qua trinh train, giup phat hien divergence som.
- Neu chi co 1 epoch, can them it nhat 1-2 epoch de quan sat xu huong overfit/underfit.
- Neu muon so sanh cong bang hon, giu nguyen alpha hoac thu sweep (64/128) de tach anh huong quantization.

## Appendix: re-run eval (optional)

```bash
PYTHONPATH=. python src/app/eval_cpt.py --config configs/qwen3_4b_cpt_lora.yaml --adapter outputs/qwen3-4b-cpt-lora/adapter_final
PYTHONPATH=. python src/app/eval_cpt.py --config configs/qwen3_4b_cpt_qlora.yaml --adapter outputs/qwen3-4b-cpt-qlora/adapter_final
```
