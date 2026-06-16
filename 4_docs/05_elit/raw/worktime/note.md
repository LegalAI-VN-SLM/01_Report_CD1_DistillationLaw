GVHD	Thầy Lê Anh Cường	
Email GVHD	leanhcuong@tdtu.edu.vn	leanhcuong@gmail.com
Tên HV	Hoàng Đình Quý Vũ	
SDT	`0868245465	
Email HV: 	252805008@student.tdtu.edu.vn	
		
Email	hoangdinhquyvu.snape.22@gmail.com	

---
Tên title mới

A Hybrid Knowledge Distillation and Preference Optimization Framework for Vietnamese Legal Reasoning in Small Language Models.

"Khung tích hợp Chưng cất tri thức và Tối ưu hóa sở thích cho tác vụ Suy luận pháp lý Việt Nam trên các Mô hình ngôn ngữ nhỏ."

-------------

16/03 - 22/03	Lên kế hoạch & Cơ sở lý thuyết	"- Hoàn thiện đề cương chi tiết: domain liên quan đến luật.
- Tìm đọc các paper liên quan lĩnh vực distillation model và law, SLM. 
- Tóm tắt và hiểu được các khái niệm và tìm hiểu các data để chuẩn bị cho tuần tiếp theo
"
23/03 - 29/03	Thu thập dữ liệu & Chạy Baseline	"Chạy Baseline        
- Tải tập dữ liệu: Subset 50k văn bản luật, VLQA và VLegal-Bench Level 5 (250 sample bias).
- Tiền xử lý dữ liệu: Làm sạch văn bản, chunking theo điều luật, chuẩn hóa format HuggingFace.
- Tải mô hình Student: Qwen3-4B hoặc PhoGPT-7B (4-bit) bằng Unsloth để test môi trường.
- Chạy Baseline: Đánh giá Student chưa fine-tune trên VLegal-Bench lấy điểm gốc (Accuracy, Perplexity)."
30/03 - 05/04	Sinh dữ liệu tổng hợp (Synthetic)	"- Thiết lập kết nối với Teacher (CMC-AI-Legal-32B) qua API hoặc HuggingFace.
- Viết script prompt mồi để sinh Q&A pháp lý chứa định kiến giới/vùng miền.
- Chạy inference từ Teacher để tạo 50k - 100k mẫu dữ liệu tổng hợp.
- Lọc và kiểm tra thủ công 50-100 mẫu để đảm bảo không lỗi format/kiến thức luật."
06/04 - 12/04	Huấn luyện Adapter (LoRA)	"Cấu hình LoRA (r, alpha, target modules) bằng PEFT cho mô hình Student.
- Chạy SFT lớp Adapter trên 50k dữ liệu luật cơ bản bằng Colab T4 (1-2 epochs).
- Lưu trọng số Adapter (~100MB).
- Test thử câu hỏi pháp lý để kiểm tra khả năng thích nghi văn phong luật Việt Nam."
13/04 - 19/04	Báo cáo giữa kỳ & Khởi tạo KD	"- Nộp báo cáo tiến độ: Mô tả data, mã nguồn và kết quả Baseline/LoRA bước đầu.
- Viết script huấn luyện Distillation: Setup hàm loss (KD_loss + CE_loss).
- Áp dụng Pruning: Dùng Wanda cắt tỉa ~50% sparsity để làm nhẹ mô hình Student.
- Chuẩn bị GPU mạnh (thuê theo giờ) cho quá trình chưng cất chính thức."
20/04 - 26/04	Thực hiện Distillation	"- Khởi chạy Distillation: Dùng tập synthetic (Tuần 14) khớp đầu ra Student với logits của Teacher.
- Giám sát huấn luyện: Theo dõi log/loss chart, điều chỉnh Hyper-parameters tránh overfit
- Lưu checkpoint mô hình Student sau khi chưng cất.
- Test nhận diện định kiến: Đảm bảo phục hồi ~90% hiệu năng của Teacher."
27/04 - 03/05	Lượng tử hóa (Quantization)	"- Merge trọng số LoRA Adapter vào mô hình Student gốc.
- Dùng AWQ/GPTQ lượng tử hóa về 4-bit (giảm size xuống ~2GB).
- Test lỗi suy giảm trí tuệ: Đảm bảo độ chính xác không giảm đột ngột sau nén.
- Đo lường Latency: Chạy thử trên CPU/Laptop cá nhân kiểm tra tốc độ sinh chữ."
04/05 - 10/05	Đánh giá mô hình (Evaluation)	"- Chạy đánh giá tự động trên VLegal-Bench Level 5 (Mục tiêu: Acc > 75%, F1 > 0.75).
- Trích xuất biểu đồ so sánh: Baseline vs Distilled vs Teacher.
- Thực hiện Human Test: Nhờ bạn bè/luật sư đánh giá thủ công câu trả lời của AI.
- Lưu trữ logs, bảng biểu và ảnh chụp màn hình cho báo cáo."
11/05 - 17/05	Đóng gói & Triển khai Demo	"- Xây dựng giao diện Web UI đơn giản bằng Streamlit hoặc Gradio.
- Tích hợp mô hình 4-bit: Chức năng tô đậm/báo cáo câu chứa định kiến trong hợp đồng.
- Thêm Disclaimer pháp lý: ""AI không thay thế luật sư"" trên giao diện.
- Kiểm tra độ ổn định của link demo (văn bản dài/ngắn, lỗi server)."
18/05 - 24/05	Viết nháp báo cáo chuyên đề	"- Viết Chương 1 & 2: Mở đầu, lý thuyết (KD, PDQ, định kiến trong luật VN).
- Viết Chương 3: Phương pháp thực hiện (Kiến trúc mô hình, quy trình tạo data).
- Viết Chương 4: Kết quả thực nghiệm (Biểu đồ Acc/F1, hình ảnh Demo).
- Chuẩn hóa danh mục tài liệu tham khảo theo quy định của trường."
25/05 - 31/05	Hoàn thiện tài liệu & Slide	"- Hoàn thiện file PDF báo cáo cuối cùng và rà soát đạo văn.
- Viết file README.md hướng dẫn cài đặt code chi tiết cho GVHD
- Thiết kế Slide thuyết trình (tập trung vào PDQ và kết quả thực nghiệm)
- Quay video demo hệ thống (3-5 phút) upload Drive/YouTube."
01/06 - 07/06	Nộp báo cáo có xác nhận đồng ý của GVHD	"- 01/ 06-08/ 06: Nộp báo cáo có xác nhận đồng ý của GVHD
- Bộ môn phân công sắp xếp hội đồng phản biện
- Có thể tinh chỉnh thêm để hoàn thiện hơn"