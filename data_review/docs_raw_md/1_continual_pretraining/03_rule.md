Với CPT (Continual Pre-Training), chỉ số chính là:

---

## Trong lúc training

**Perplexity (PPL)**
Chỉ số chuẩn cho language modeling. Công thức: `PPL = exp(cross-entropy loss)`.

- PPL thấp = model dự đoán text tốt hơn
- Theo dõi trên **held-out set** — tách ~5% data ra, không train, chỉ dùng để đo
- Kỳ vọng: PPL trên legal text phải **giảm** sau CPT so với base model

---

## Sau training — đánh giá thực chất

CPT không trực tiếp giải quyết task nào, nên cần đánh giá gián tiếp qua downstream:

| Cách đánh giá | Làm gì | Kỳ vọng |
|---|---|---|
| **Legal QA** | Hỏi câu hỏi về luật, so sánh câu trả lời | CPT model trả lời chính xác hơn base |
| **Cloze test** | Che 1 từ trong câu luật, model đoán | Accuracy cao hơn trên legal domain |
| **SFT rồi so sánh** | Fine-tune cả base và CPT model trên cùng SFT data | CPT model converge nhanh hơn, PPL thấp hơn |

---

## Thực tế với dataset của bạn (~46M tokens)

Dataset khá nhỏ cho CPT → không kỳ vọng PPL giảm mạnh, nhưng đủ để:
- Model quen vocabulary pháp lý VN (tên luật, điều khoản, cơ quan)
- Giảm hallucination trên legal domain khi SFT

**Bước cụ thể nên làm:**
```
1. Tách 5% data → eval set (dùng train_test_split)
2. Train CPT → log perplexity trên eval set mỗi N steps
3. So sánh PPL của base model vs CPT model trên cùng eval set
```

Bạn muốn mình viết script tách eval set luôn không?

---

