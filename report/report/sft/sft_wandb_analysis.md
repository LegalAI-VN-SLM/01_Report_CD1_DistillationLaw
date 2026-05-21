# SFT W&B + Eval Report (legal-slm-sft-v1)

Generated: 2026-05-21

## Nguon du lieu

- data_wandb/SFT/legal-slm-sft-v1 (config.json, history.csv, summary.json)
- data_evaluate/SFT_lora_gwen3-1.7b-pretrain (summary.json, mc_results.json, nli_results.json, syllogism_results.json)

## Thong tin run

- run_name: legal-slm-sft-v1
- run_id: 83i6hlga
- state: finished

## Cau hinh train (tom tat)

- Base model: VLSP2025-LegalSML/qwen3-1.7b-legal-pretrain
- Dtype: bf16 (bfloat16), gradient checkpointing: true
- LoRA: r=16, alpha=32, dropout=0.05, target_modules=[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
- Quantization: 4-bit NF4 (bnb_4bit_compute_dtype=bfloat16, double_quant=true)
- Seq length: max_seq_length=2048, max_position_embeddings=32768
- Training: epochs=3, per_device_train_batch_size=16, grad_accum=2, lr=2e-4, scheduler=cosine, warmup_ratio=0.1, weight_decay=0.01
- Eval/Log: eval_steps=200, save_steps=200, logging_steps=10
- Packing: false, padding_free: true

## Tom tat training (wandb summary)

| Metric | Gia tri |
| --- | --- |
| train/loss (last) | 0.5220 |
| train_loss (aggregated) | 0.6730 |
| eval/loss (last) | 0.7462 |
| train/global_step | 489 |
| train/epoch | 3 |
| train_runtime (s) | 1246.33 |
| train_samples_per_second | 12.531 |
| train_steps_per_second | 0.392 |
| eval_steps_per_second | 2.165 |
| eval_samples_per_second | 34.001 |

### Xu huong tu history.csv

- Train loss: start 1.5037 -> min 0.4703 -> last 0.5220
- Eval loss (theo step): 200=0.7612, 400=0.7437, 489=0.7462

## Ket qua danh gia 3 task

### Tong quan (overall)

| Task | Metric | Gia tri | n | Unknown |
| --- | --- | --- | --- | --- |
| NLI | accuracy / f1 | 0.6809 / 0.4819 | 141 | 16 |
| MC | accuracy | 0.1600 | 100 | 64 |
| Syllogism | rouge1 / rouge2 / rougeL | 0.4444 / 0.3566 / 0.3897 | 84 | n/a |

### Theo thinking flag (by_thinking)

| Task | thinking=true | n | thinking=false | n |
| --- | --- | --- | --- | --- |
| NLI (acc / f1) | 0.6509 / 0.4606 | 106 | 0.7714 / 0.5466 | 35 |
| MC (acc) | 0.1600 | 75 | 0.1600 | 25 |
| Syllogism (r1 / r2 / rL) | 0.4381 / 0.3535 / 0.3823 | 63 | 0.4635 / 0.3659 / 0.4118 | 21 |

## Vi du per_sample (minh hoa)

### MC (sai/khong co dap an)

| id | thinking | gold | pred | ghi chu |
| --- | --- | --- | --- | --- |
| vilawqa_mc_00593 | true | B | null | missing answer |
| vilawqa_mc_00960 | true | A | null | missing answer |
| vilawqa_mc_00314 | true | A | null | missing answer |
| vilawqa_mc_00806 | true | D | A | wrong answer |
| vilawqa_mc_00562 | true | B | C | wrong answer |

### NLI (unknown/sai)

| id | thinking | gold | pred | ghi chu |
| --- | --- | --- | --- | --- |
| vilawqa_nli_01073 | true | relevant | unknown | unknown label |
| vilawqa_nli_00396 | false | relevant | unknown | unknown label |
| vilawqa_nli_00851 | true | relevant | not_relevant | wrong label |
| vilawqa_nli_00662 | true | not_relevant | relevant | wrong label |
| vilawqa_nli_00013 | false | not_relevant | relevant | wrong label |

### Syllogism (best/worst by rougeL)

**Top 3**

| id | thinking | rouge1 | rouge2 | rougeL |
| --- | --- | --- | --- | --- |
| vilawqa_syllo_00488 | false | 0.8250 | 0.8235 | 0.8250 |
| vilawqa_syllo_00277 | true | 0.8851 | 0.8475 | 0.8108 |
| vilawqa_syllo_00416 | true | 0.7859 | 0.7797 | 0.7859 |

**Bottom 3**

| id | thinking | rouge1 | rouge2 | rougeL |
| --- | --- | --- | --- | --- |
| vilawqa_syllo_00449 | true | 0.0000 | 0.0000 | 0.0000 |
| vilawqa_syllo_00399 | true | 0.0000 | 0.0000 | 0.0000 |
| vilawqa_syllo_00242 | true | 0.0076 | 0.0000 | 0.0076 |

## Nhan xet nhanh

- Train loss giam on dinh; eval loss dao dong nhe quanh ~0.74, khong co dau hieu overfit ro rang trong 3 epoch.
- NLI dat accuracy ~0.68, F1 ~0.48; nhom thinking=false tot hon thinking=true (acc +0.12, f1 +0.09).
- MC rat thap (0.16) va unknown cao (64%), goi y van de dinh dang output hoac extract dap an.
- Syllogism o muc trung binh (rougeL ~0.39); thinking=false cao hon thinking=true.
- Nhieu du lieu co pred null/unknown cho MC/NLI, can kiem tra format output va bo loc/parse.

## Khuyen nghi

- Kiem tra prompt va parser cho MC/NLI; ap dung constrained decoding hoac regex extract dap an de giam unknown/null.
- Tach biet so sanh thinking vs direct voi cung prompt; neu direct tot hon, can giu che do direct khi suy luan.
- Bo sung data hoac re-weight cho MC (multiple-choice) neu day la muc tieu chinh.
- Neu can theo doi xu huong, tao plot train/loss va eval/loss tu history.csv de quan sat dao dong chi tiet.
