Optimizing tool selection...Đây là ba nhãn (label) chuẩn trong tác vụ **Natural Language Inference (NLI)** - suy diễn ngôn ngữ tự nhiên:

1. **Entailment** (Suy diễn): Giả thuyết được suy ra logic từ tiền đề. Nếu tiền đề đúng, giả thuyết chắc chắn đúng.
   - *Ví dụ*: 
     - Tiền đề: "Một người đàn ông đang chạy trên bãi biển"
     - Giả thuyết: "Có một người đàn ông"
     - → **Entailment** ✓

2. **Neutral** (Trung lập): Không thể xác định giả thuyết đúng hay sai từ tiền đề. Giữa chúng không có mối quan hệ rõ ràng.
   - *Ví dụ*:
     - Tiền đề: "Một người đàn ông đang chạy trên bãi biển"
     - Giả thuyết: "Anh ta đang mặc áo xanh"
     - → **Neutral** (không đủ thông tin)

3. **Contradiction** (Mâu thuẫn): Giả thuyết mâu thuẫn trực tiếp với tiền đề. Nếu tiền đề đúng, giả thuyết sai.
   - *Ví dụ*:
     - Tiền đề: "Một người đàn ông đang chạy trên bãi biển"
     - Giả thuyết: "Người đàn ông đang ngồi yên"
     - → **Contradiction** ✗

Trong dự án của bạn, tác vụ này được sử dụng ở `vlsp-2023-full/vilawqa-nli/` cho các câu hỏi liên quan đến luật pháp Việt Nam.