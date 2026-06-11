# Từ Điển Thuật Ngữ Chi Tiết V2 — Dự Án Legal SLM Distillation

Tài liệu này là phiên bản nâng cấp, mở rộng toàn diện các thuật ngữ kỹ thuật, thuật toán chưng cất, phương pháp giải mã và các độ đo đánh giá (evaluation metrics) được sử dụng xuyên suốt dự án **Legal SLM Distillation (V1 - V4)**. Các định nghĩa được thiết kế ngắn gọn, chuẩn xác học thuật để người viết có thể trích dẫn trực tiếp vào luận văn/bài báo khoa học.

---

## 📂 1. Thuật Ngữ Hệ Thống & Huấn Luyện (System & Training Terms)

| Thuật ngữ | Khái niệm gốc | Giải thích chi tiết trong dự án |
| :--- | :--- | :--- |
| **SFT Baseline** | Supervised Fine-Tuning Baseline | Mô hình nền tảng được huấn luyện bằng phương pháp tinh chỉnh có giám sát trực tiếp trên tập dữ liệu luật pháp. Đây là điểm xuất phát để chưng cất tri thức hoặc so sánh hiệu năng. |
| **Teacher-Student Gap** | Khoảng cách Giáo viên - Học sinh | Sự chênh lệch lớn về năng lực hoặc kích thước tham số giữa mô hình Teacher (4B GRPO) và Student (0.6B/1.7B). Khoảng cách quá lớn có thể khiến Student không thể hội tụ nếu ép học theo phân phối quá phức tạp của Teacher. |
| **Catastrophic Forgetting** | Quên lãng thảm họa | Hiện tượng mạng nơ-ron đột ngột mất đi hoàn toàn các khả năng/định dạng đã học ở bước SFT (như khả năng suy nghĩ bằng tiếng Việt, định dạng `<think>`) sau khi huấn luyện KD với tham số chưa tối ưu (LR quá cao). |
| **Alignment Tax** | Thuế căn chỉnh | Sự đánh đổi về hiệu năng, trong đó mô hình sau khi được căn chỉnh hành vi (bằng DPO/RLHF) có thể bị sụt giảm điểm số ở một số tác vụ logic hoặc ngôn ngữ tự nhiên cơ bản để đổi lấy sự an toàn hoặc định dạng đầu ra mong muốn. |
| **Effective Batch Size** | Kích thước lô hiệu dụng | Số lượng mẫu thực tế được tính toán gradient trước khi cập nhật trọng số mô hình: $Batch\ Size \times Gradient\ Accumulation \times GPU\ Count$. Trong dự án, tăng effective batch size từ 16 lên 64 giúp ổn định hướng đi của gradient, giảm nhiễu. |
| **QLoRA Regularization** | Bộ điều chuẩn ngầm QLoRA | Việc đóng băng trọng số base ở định dạng 4-bit NF4 và chỉ cập nhật LoRA adapters nhỏ. Cơ chế này hoạt động như một bộ điều chỉnh ngầm, bảo vệ tri thức SFT gốc không bị ghi đè thô bạo bởi tín hiệu KD. |
| **SFT Anchor (CE Anchor)** | Điểm neo SFT | Việc duy trì một phần trọng số Cross-Entropy Loss ($L_{CE}$) trên nhãn cứng gốc (gold labels) trong hàm loss hỗn hợp nhằm giữ chân mô hình không trôi dạt quá xa khỏi dữ liệu chuẩn. |

---

## 📂 2. Kỹ Thuật Chưng Cất & Căn Chỉnh Nâng Cao (Advanced Distillation & Alignment)

| Thuật ngữ | Khái niệm gốc | Giải thích chi tiết trong dự án |
| :--- | :--- | :--- |
| **Exposure Bias** | Thiên kiến phơi nhiễm | Lỗi lệch phân phối xảy ra khi mô hình Học sinh (Student) tự sinh văn bản lúc suy luận (inference), sau khi chỉ được huấn luyện bằng cách bắt chước phân phối của Teacher trên các đáp án chuẩn của tập huấn luyện (training). |
| **On-Policy KD** | On-Policy Knowledge Distillation | Phương pháp chưng cất trong đó Student tự sinh ra các câu trả lời của mình (rollouts), sau đó Teacher mới chấm điểm và tính loss KL trên các chuỗi này, giúp Student tự sửa sai trên chính quỹ đạo phân phối của mình. |
| **Privileged Information (PI)** | Thông tin đặc quyền | Kỹ thuật cung cấp sẵn đáp án đúng (Gold Answer) trong prompt của Teacher để Teacher đóng vai trò người chấm thi toàn tri (Oracle), giúp sinh ra logits chấm điểm chính xác và có độ tự tin cao nhất. |
| **KL Divergence** | Kullback-Leibler Divergence | Độ đo sự khác biệt giữa hai phân phối xác suất. Trong KD, nó được dùng làm hàm loss để ép phân phối xác suất của Student tiệm cận phân phối của Teacher. |
| **Forward-KL** | Forward KL Divergence | Công thức tính $D_{KL}(P_{teacher} \parallel P_{student})$. Hướng tới việc bao phủ toàn bộ không gian xác suất của Teacher (zero-avoiding), dễ làm Student sinh từ ngữ mơ hồ nếu năng lực mô hình nhỏ không đủ đáp ứng. |
| **Reverse-KL** | Reverse KL Divergence | Công thức tính $D_{KL}(P_{student} \parallel P_{teacher})$. Hướng tới việc tìm các chế độ xác suất cao nhất của Teacher (mode-seeking), giúp Student sinh câu trả lời sắc nét, tập trung và bớt lan man hơn. |
| **KL Clipping** | Cắt cụt KL Divergence | Cơ chế giới hạn giá trị KL loss tại từng token ở một ngưỡng $C$ (ví dụ: $5.0$) để ngăn chặn các token cấu trúc XML (`<think>`, `<major_premise>`) có mức chênh lệch lớn chiếm lĩnh toàn bộ gradient. |
| **Top-K Sparse Logits** | Logits thưa Top-K | Kỹ thuật trích xuất và lưu trữ chỉ Top-K logits lớn nhất (ví dụ: $K=50$) từ Teacher thay vì lưu toàn bộ từ vựng (vocab ~151k), giúp giảm 90% dung lượng ổ cứng khi cache logits. |
| **DPO Alignment** | Direct Preference Optimization | Giai đoạn tinh chỉnh cuối cùng sử dụng cặp dữ liệu phản hồi tốt (Chosen) và kém (Rejected) nhằm giúp mô hình học sinh tự điều chỉnh hành vi, giảm ảo tưởng và bám sát định dạng suy luận. |
| **Diagnosis-Driven DPO** | DPO hướng chẩn đoán | Quy trình xây dựng tập preference cho DPO dựa trên việc chẩn đoán lỗi sai thực tế của Student thông qua các nhãn phân loại (OK, WRONG, RISKY, PARTIAL) của LLM Judge. |

---

## 📂 3. Giải Mã & Đánh Giá Hiệu Năng (Decoding & Evaluation)

| Thuật ngữ | Khái niệm gốc | Giải thích chi tiết trong dự án |
| :--- | :--- | :--- |
| **Greedy Decoding** | Giải mã tham lam | Cơ chế sinh text lấy token có xác suất cao nhất tại mỗi bước giải mã ($do\_sample=False$). Được sử dụng bắt buộc trong giai đoạn **Đánh giá (Evaluation)** để bảo đảm tính nhất quán, khách quan và khả năng tái lập kết quả. |
| **Sampling Decoding** | Giải mã lấy mẫu | Cơ chế sinh text lấy mẫu ngẫu nhiên từ phân phối xác suất ($do\_sample=True$, kèm $Temperature$, $top\_p$). Dùng trong huấn luyện **On-Policy KD** và thu thập dữ liệu **DPO** để tạo ra các quỹ đạo suy luận đa dạng, giúp khám phá các lỗi sai của Student. |
| **Temperature ($T$)** | Nhiệt độ giải mã / chưng cất | 1. *Trong giải mã:* Điều khiển độ ngẫu nhiên (T thấp $\to$ chính xác/greedy; T cao $\to$ sáng tạo/đa dạng).<br>2. *Trong chưng cất:* Làm mịn phân phối logits của Teacher trước khi tính KL ($T$ càng lớn, phân phối càng phẳng, truyền đạt nhiều "tri thức tối" - dark knowledge hơn). |
| **LLM-as-a-Judge** | Mô hình lớn làm giám khảo | Việc sử dụng một LLM mạnh (GPT-4o-mini) làm giám khảo đánh giá ngữ nghĩa và logic của câu trả lời, khắc phục nhược điểm của Regex (bỏ sót đáp án đúng ngữ nghĩa nhưng sai format) và ROUGE-L (vô tình chấm điểm cao cho câu trả lời sai logic nhưng trùng từ khóa). |
| **Macro F1-Score** | Điểm F1 trung bình Macro | Chỉ số đánh giá trung bình cộng F1-score của các lớp phân loại một cách bình đẳng, không phụ thuộc số lượng mẫu. Độ đo chính cho tác vụ **NLI** để kiểm soát hiệu năng trên cả mẫu liên quan và không liên quan. |
| **ROUGE-L** | Longest Common Subsequence | Độ đo đánh giá dựa trên chuỗi con chung dài nhất giữa câu trả lời sinh ra và nhãn chuẩn. Được dùng để đánh giá mức độ chính xác của câu trả lời cuối cùng trong tác vụ **Tam đoạn luận (Syllogism)**. |
| **Overall Score** | Điểm số tổng hợp | Chỉ số trung bình cộng hiệu năng của 3 tác vụ chính: $Overall = (NLI\ Acc + MC\ Acc + Syllogism\ ROUGE\text{-}L) / 3$. Đây là thước đo hiệu năng tổng thể của SLM trong miền luật pháp Việt Nam. |

---

## 📂 4. Các Tác Vụ Đánh Giá Pháp Lý (Legal Evaluation Tasks)

*   **NLI (Natural Language Inference - Nhận định Luật):** Tác vụ phân loại logic xác định mối quan hệ giữa một văn bản pháp luật và một nhận định cụ thể. Rút gọn thành 2 lớp: **Có liên quan** (Entailment) hoặc **Không liên quan** (Not Relevant).
*   **MC (Multi-choice - Trắc nghiệm pháp luật):** Tác vụ lựa chọn một đáp án đúng duy nhất `[A-D]` dựa trên câu hỏi và các điều luật đi kèm. KD giúp Student tận dụng "tri thức mềm" của Teacher để loại trừ các đáp án nhiễu tốt hơn.
*   **Syllogism (Tam đoạn luận pháp lý):** Tác vụ suy luận logic phức tạp nhất, yêu cầu SLM phải lập luận qua 3 bước rõ ràng:
    1.  *Đại tiền đề (Major Premise):* Trích xuất chính xác điều luật chung.
    2.  *Tiểu tiền đề (Minor Premise):* Tóm tắt hành vi/sự kiện thực tế của vụ việc.
    3.  *Kết luận (Conclusion):* Đưa ra phán quyết pháp lý cuối cùng.
    
    Yêu cầu mô hình phải suy nghĩ bên trong thẻ `<think> ... </think>` trước khi đưa ra kết luận.
