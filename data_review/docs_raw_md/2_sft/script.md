# Script Presentation — Supervised Fine-Tuning

---

## Slide 1: Title

> Chào mọi người. Phần này mình sẽ trình bày về **Supervised Fine-Tuning** — bước tiếp theo sau Continual Pre-Training trong pipeline xây dựng Legal SLM.
>
> Nếu CPT là dạy model đọc văn bản luật, thì SFT là dạy model **trả lời câu hỏi** pháp luật. Đây là bước biến model thành trợ lý pháp lý thực sự.

---

## Slide 2: CPT → SFT — Bước chuyển then chốt

> Sau CPT, model đã quen vocabulary pháp lý — nó biết "nghị định", "điều khoản", "cơ quan ban hành" là gì. Nhưng nó chưa biết cách **nhận câu hỏi và đưa ra câu trả lời**.
>
> SFT giải quyết bằng cách train trên dữ liệu dạng cặp **instruction - response**. Ví dụ: "Với câu hỏi về Luật Hải quan, hãy trả lời như thế này." Model học cách follow instruction, không chỉ dự đoán token nữa.
>
> Nói đơn giản: CPT cho model biết ngôn ngữ luật, SFT cho model biết cách dùng kiến thức đó để trả lời.

---

## Slide 3: Cấu hình training

> Về cấu hình training. Mình dùng **Qwen3-1.7B** làm base model, quantize 4-bit NF4 bằng QLoRA.
>
> Batch size 16, gradient accumulation 2, nên effective batch là **32 samples** mỗi lần update weights. Learning rate 2e-4 với cosine scheduler — tức LR sẽ tăng dần trong 10% steps đầu rồi giảm dần theo đường cosine về gần 0.
>
> Optimizer dùng **adamw_8bit** — tiết kiệm khoảng 50% VRAM cho optimizer states. Kết hợp gradient checkpointing tiết kiệm thêm 60% VRAM. Kết quả: chỉ dùng khoảng **6GB trên GPU 24GB** — còn dư rất nhiều, lần sau có thể tăng batch.

---

## Slide 4: LoRA — Chỉ train 2% tham số

> Điểm đặc biệt của LoRA: thay vì sửa toàn bộ 1.7 tỷ tham số, mình chèn 2 ma trận nhỏ A và B vào mỗi layer. Rank 16, alpha 32, scaling factor = 2.0.
>
> Target 7 loại module: 4 attention projections (q, k, v, o) và 3 MLP layers (gate, up, down). Kết quả: adapter file chỉ nặng khoảng **50MB** thay vì 3.5GB full model.
>
> Quan trọng: giữ nguyên weights gốc nên model **không mất kiến thức** đã có từ pre-training và CPT. LoRA chỉ thêm khả năng mới lên trên.

---

## Slide 5: Đọc WandB Dashboard

> Trong lúc training, WandB hiển thị realtime các chỉ số. Hai cái quan trọng nhất:
>
> **train/loss** phải giảm liên tục — nếu không là model không đủ capacity hoặc LR quá thấp. Dưới 0.6 là tốt, trên 2.0 là chưa học được gì.
>
> **eval/loss** là chỉ số quan trọng nhất. Nó phải giảm cùng train/loss. Nếu eval/loss **bắt đầu tăng** trong khi train/loss vẫn giảm — đó là dấu hiệu **overfit** rõ ràng, cần dừng sớm.
>
> grad_norm nên ở khoảng 0.1 đến 1.0, trên 10 là gradient exploding.

---

## Slide 6: 3 Task đánh giá

> Đánh giá model trên dataset **vilawqa** — Vietnamese Legal QA — gồm 3 loại bài từ đơn giản đến phức tạp.
>
> **NLI**: 188 samples, binary classification. Cho đoạn văn bản luật và câu hỏi, model phán đoán "Có liên quan" hay "Không liên quan". Baseline là 55-60% nếu predict có liên quan hết. Target: trên **85%**.
>
> **Multi-choice**: 133 samples, chọn 1 trong 4 đáp án A/B/C/D. Random baseline 25%. Target: trên **70%**.
>
> **Syllogism**: 112 samples, open-ended — trả lời câu hỏi pháp lý dựa trên căn cứ. Không có đáp án cố định, đo bằng **ROUGE-L**, target trên 0.40. Lưu ý ROUGE thấp không nhất thiết là sai — model có thể paraphrase đúng nhưng khác wording với reference.

---

## Slide 7: Thinking Mode

> Qwen3 có feature hay là **Thinking Mode**. Hai chế độ:
>
> **Thinking ON**: inject prefix `<think>` vào sau prompt, model sẽ tự viết reasoning chain — phân tích điều luật, suy luận — rồi mới đưa ra câu trả lời. Chậm hơn nhưng **chính xác hơn** cho câu hỏi phức tạp.
>
> **Thinking OFF**: inject tag think rỗng, model bỏ qua reasoning và trả lời thẳng. Nhanh, phù hợp câu hỏi đơn giản.
>
> Test set có 75% samples có thinking=True và 25% thinking=False. Khi tính metrics, phần think bị strip đi — chỉ đánh giá câu trả lời cuối cùng.
>
> **Quan trọng**: cần chạy eval cả 2 mode. Nếu accuracy thinking cao hơn hẳn direct — model phụ thuộc CoT, không trả lời thẳng được. Ngược lại thì model ổn định, không cần CoT.

---

## Slide 8: Đọc kết quả — Cross-task patterns

> Đọc kết quả không chỉ nhìn từng con số riêng lẻ. Cần đọc **cross-task patterns** để hiểu model thiếu gì:
>
> **NLI thấp mà MC cao** — model biết facts, nhớ đáp án, nhưng không phân biệt được điều luật nào liên quan đến câu hỏi nào. Thiếu khả năng relevance matching.
>
> **MC thấp mà Syllogism ROUGE cao** — model viết câu trả lời nghe hay, paraphrase tốt, nhưng logic reasoning yếu. Chọn đáp án sai vì không suy luận được.
>
> Breakdown by thinking cũng rất quan trọng: nếu accuracy khi bật thinking cao hơn hẳn khi tắt, thì nên bật CoT khi deploy. Ngược lại, model đã ổn định và trả lời thẳng tốt hơn.

---

## Slide 9: Thank you

> Tóm lại pipeline SFT: **Qwen3-1.7B** quantize 4-bit, train bằng LoRA rank 16, eval trên 3 task NLI, Multi-choice, Syllogism.
>
> Chạy cả 2 thinking mode để biết khi nào nên bật suy nghĩ. Kết quả cho biết model mạnh ở đâu — facts, reasoning, hay relevance — và từ đó chọn mode deploy phù hợp.
>
> Cảm ơn mọi người. Mình sẵn sàng nhận câu hỏi.

---

## Phím tắt Presentation Mode

| Phím | Chức năng |
|------|-----------|
| `→` / `Space` | Slide tiếp |
| `←` | Slide trước |
| `N` | Bật/tắt Speaker Notes |
| `F` | Fullscreen |
| `Esc` | Quay về Story Mode |
