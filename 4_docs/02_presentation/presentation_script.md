# Script Thuyết Trình — Capstone Project 1
### "A Multi-Phase Knowledge Distillation Framework for Vietnamese Legal Reasoning in Small Language Models"
**Người trình bày:** Hoàng Đình Quý Vũ — 252805008 · **GVHD:** Assoc. Prof. Dr. Lê Anh Cường

> Cách dùng: phần **🎤 Nói** là lời thoại. Phần *→ chuyển ý* là câu bắc cầu sang slide sau (đọc liền mạch). Mục tiêu 15 slide chính ≈ **10 phút**. Slide 16–23 là **backup** — chỉ lật khi hội đồng hỏi.

---

## PHẦN 1 — BÀI NÓI CHÍNH (Slide 1–15, ~10 phút)

### 🟦 Slide 1 — Title *(~20s)*
🎤 "Kính chào quý thầy cô trong hội đồng. Em là Hoàng Đình Quý Vũ, sinh viên Khoa Công nghệ Thông tin, Trường Đại học Tôn Đức Thắng. Hôm nay em xin trình bày đồ án Capstone Project 1 với đề tài *Khung chưng cất tri thức đa pha cho suy luận pháp lý tiếng Việt trên các Mô hình Ngôn ngữ Nhỏ*, dưới sự hướng dẫn của thầy Lê Anh Cường."

*→ chuyển ý:* "Trước khi đi vào chi tiết, em xin tóm tắt nhanh nội dung."

---

### 🟦 Slide 2 — Table of Contents *(~20s)*
🎤 "Bài của em gồm bốn mạch: **một** — đặt vấn đề và mục tiêu; **hai** — kiến trúc pipeline và ba pha chưng cất; **ba** — thiết lập thí nghiệm; và **bốn** — kết quả, ablation và giới hạn. Em sẽ dừng lâu nhất ở phần kết quả."

*→ chuyển ý:* "Bắt đầu từ câu hỏi: vì sao bài toán này quan trọng?"

---

### 🟦 Slide 3 — Problem & Motivation *(~55s)*
🎤 "Các mô hình ngôn ngữ nhỏ, dưới 2 tỷ tham số, rất hấp dẫn cho lĩnh vực pháp lý: độ trễ thấp, chi phí thấp, và quan trọng nhất là **chạy hoàn toàn tại chỗ** — hồ sơ vụ án không rời khỏi tổ chức.

Nhưng mô hình nhỏ thì **khó suy luận**. Có hai khó khăn: thứ nhất là **khoảng cách dung lượng** — một mô hình 0.6 đến 1.7 tỷ tham số không thể sánh với mô hình thầy lớn đã được căn chỉnh; thứ hai là **thiên lệch phơi nhiễm** (exposure bias) — lỗi sớm trong chuỗi suy luận sẽ tích lũy dọc theo chính chuỗi sinh của học sinh.

Hướng tiếp cận của em: **chưng cất khả năng suy luận của một mô hình thầy 4 tỷ tham số đã căn chỉnh bằng GRPO, vào một mô hình học sinh nhỏ gọn, có thể triển khai tại chỗ**."

*→ chuyển ý:* "Vấn đề rõ rồi, nhưng khoảng trống nghiên cứu nằm ở đâu?"

---

### 🟦 Slide 4 — Research Gap & Objectives *(~50s)*
🎤 "Có ba khoảng trống. Một — chưng cất offline tiêu chuẩn **bỏ qua exposure bias** trong suy luận pháp lý nhiều bước. Hai — dữ liệu sở thích cho căn chỉnh thường được lọc bằng **luật cứng dễ vỡ**, vô tình thưởng cho các 'đoán may' và loại bỏ câu đúng nhưng khác định dạng. Ba — rất ít nghiên cứu khảo sát các phương pháp này ở **quy mô dưới 2 tỷ** cho tiếng Việt.

Từ đó em đặt năm mục tiêu: thiết kế **khung chưng cất đa pha**; chưng cất logit offline với **QLoRA làm bộ điều chuẩn ẩn**; dùng **on-policy KD** để giảm exposure bias; **DPO dẫn dắt bằng chẩn đoán** với LLM làm trọng tài; và đánh giá cả **in-distribution lẫn out-of-distribution** trên VLSP2025."

*→ chuyển ý:* "Năm mục tiêu này được hiện thực hóa trong một pipeline thống nhất."

---

### 🟦 Slide 5 — Proposed Pipeline *(~60s)*
🎤 "Đây là bức tranh tổng thể, và em xin nhấn mạnh một điểm hay bị hiểu nhầm: **đây không phải chuỗi tuyến tính ba pha nối tiếp**.

Em bắt đầu từ mô hình SFT, qua **Pha 1 — chưng cất logit offline** — tạo ra một **checkpoint nền chung**. Từ checkpoint chung này, **Pha 2 và Pha 3 là hai nhánh tinh chỉnh bổ sung, áp dụng độc lập**: nhánh trên là **On-Policy KD** ra mô hình On-Policy SLM; nhánh dưới là **DPO dẫn dắt bằng chẩn đoán** ra mô hình DPO-Aligned SLM.

Hai chi tiết quan trọng: mô hình thầy là **Qwen3-4B Vietnamese-legal-GRPO**; và **chính sách tham chiếu của DPO là checkpoint Pha 1** — đúng checkpoint dùng để khởi tạo Pha 2, chứ không phải đầu ra của Pha 2."

*→ chuyển ý:* "Ta đi vào từng pha. Đầu tiên là Pha 1."

---

### 🟦 Slide 6 — Phase 1 · Offline Logit Distillation *(~50s)*
🎤 "Pha 1: học sinh **bắt chước phân phối mềm** của thầy trên một tập dữ liệu tĩnh. Cùng một input đưa vào cả thầy (đóng băng, lấy Top-50 logit) và học sinh (gắn adapter QLoRA).

Hàm mất mát là tổ hợp hai phần: **Soft Loss** — KL giữa phân phối thầy và trò ở nhiệt độ T; và **Hard Loss** — cross-entropy trên token vàng. Tổng là alpha nhân soft cộng một-trừ-alpha nhân hard.

Điểm cốt lõi, và đây là một phát hiện của em: **QLoRA đóng băng base ở 4-bit và chỉ cập nhật adapter — hoạt động như một bộ điều chuẩn ẩn, ngăn quên thảm họa** ở quy mô nhỏ."

*→ chuyển ý:* "Nhưng chưng cất offline có một điểm yếu cố hữu: exposure bias. Đó là lý do có Pha 2."

---

### 🟦 Slide 7 — Phase 2 · On-Policy KD *(~55s)*
🎤 "Pha 2 giải quyết exposure bias. Thay vì học trên câu vàng cố định, học sinh **sinh ra chính rollout của nó** với decoding ngẫu nhiên — nhiệt độ 0.7, top-p 0.9.

Rollout đó được đưa cho **thầy ở chế độ Oracle** — tức là thầy có **quyền truy cập đặc biệt vào đáp án vàng y-sao**. Thầy cung cấp tín hiệu sửa lỗi ở cấp token, còn học sinh được cập nhật qua **KL có cắt ngưỡng** để tránh nổ gradient trên các đường lệch lớn.

Ý nghĩa: huấn luyện trên chính quỹ đạo tự sinh **khép lại khoảng cách giữa huấn luyện và suy luận**. Em xin thẳng thắn — việc đưa đáp án vàng cho thầy là một đường rò rỉ tiềm tàng, em có ghi rõ trong Giới hạn và sẽ nói thêm nếu hội đồng hỏi."

*→ chuyển ý:* "Nhánh còn lại từ checkpoint Pha 1 là căn chỉnh sở thích — Pha 3."

---

### 🟦 Slide 8 — Phase 3 · Diagnosis-Driven DPO *(~55s)*
🎤 "Pha 3 sửa các lỗi suy luận còn sót. Học sinh sinh rollout, rồi một **LLM Judge — GPT-4o-mini — kiểm toán ngữ nghĩa** cả chuỗi suy nghĩ lẫn kết luận cuối.

Nếu không có lỗi (OK) thì bỏ. Nếu có lỗi, mẫu được đưa qua **lấy mẫu phân tầng** tạo thành cặp sở thích: chọn là **đáp án vàng**, loại là **chính lỗi mà học sinh mắc**. Cặp này đi vào **DPO** với beta 0.1, tham chiếu là checkpoint Pha 1.

Điểm mấu chốt: **chẩn đoán ngữ nghĩa cho tập sở thích sạch hơn hẳn lọc bằng luật** — phần ablation lát nữa sẽ chứng minh bằng số."

*→ chuyển ý:* "Đó là toàn bộ phương pháp. Giờ sang thiết lập thí nghiệm."

---

### 🟦 Slide 9 — Experimental Setup & Data *(~45s)*
🎤 "Dữ liệu dựa trên bộ **ViLawQA** từ tài khoản thangvip, gồm ba tác vụ: trắc nghiệm MCQ, suy luận NLI ba lớp, và tam đoạn luận Syllogism. Tập huấn luyện 2.603 mẫu, validation 322, test nội bộ 325 — đây là phần in-distribution. Ngoài ra em dùng **VLSP2025 Public-Test 440 mẫu làm benchmark out-of-distribution**.

Thầy là Qwen3-4B legal-GRPO; trò là Qwen3 0.6B và 1.7B. Em lưu ý: điểm **Overall chỉ là trung bình không trọng số** trộn accuracy với ROUGE-L, nên nó là chỉ số tóm tắt thô — **điểm theo từng tác vụ mới là cơ sở chính** để so sánh."

*→ chuyển ý:* "Và bây giờ là kết quả — phần em muốn hội đồng lưu ý nhất."

---

### 🟦 Slide 10 — Result · Performance Across Phases *(~70s — DỪNG LÂU)*
🎤 "Đồ thị này là tiến trình điểm Overall qua từng pha, trên tập in-distribution.

Đường xanh là **1.7B — nó tinh chỉnh tốt**: sau DPO đạt **NLI cao nhất 0.8156 và Syllogism cao nhất 0.5684** trong toàn bộ thí nghiệm in-domain. Điểm Overall 0.5280 chỉ nhỉnh thấp hơn mốc offline 0.5322 — nhưng đây là **đánh đổi hồ sơ tác vụ**, không phải thất bại: DPO tối đa hóa hai tác vụ suy luận, đổi lại MCQ giảm.

Đường đỏ là **0.6B — nó sụp đổ**: on-policy và DPO khiến Overall rơi từ 0.4622 xuống 0.3104 rồi 0.1861. Bài học: **căn chỉnh cần một ngưỡng dung lượng tối thiểu** — mô hình dưới một tỷ tham số không đủ sức hấp thụ."

*→ chuyển ý:* "Vì sao em tự tin các kết luận này đến từ thiết kế chứ không phải ngẫu nhiên? Hai ablation sau đây."

---

### 🟦 Slide 11 — Ablations · QLoRA & Judge *(~55s)*
🎤 "Ablation thứ nhất, bên trái: **LoRA so với QLoRA** trên 0.6B. Đường đỏ là LoRA — cross-entropy loss **tăng từ 0.3 lên 0.7**, dấu hiệu quên thảm họa, Syllogism chỉ 0.1741. Đường xanh là QLoRA — loss **giảm mượt từ 0.5 xuống 0.25**, Syllogism vọt lên 0.3804. Chứng minh QLoRA thực sự là bộ điều chuẩn.

Ablation thứ hai, bên phải: **luật cứng so với LLM Judge** trên 2.603 rollout. Judge ngữ nghĩa **phục hồi các câu đúng-nhưng-khác-định-dạng**, nâng OK từ 11% lên 21%; và **vạch trần các đoán may**, nâng phát hiện WRONG từ 41% lên 64%. Kết quả là tập sở thích sạch hơn nhiều cho DPO."

*→ chuyển ý:* "Câu hỏi cuối: các mô hình này tổng quát hóa ra dữ liệu lạ thế nào?"

---

### 🟦 Slide 12 — Result · OOD Generalization *(~55s)*
🎤 "Đây là tập VLSP2025 — hoàn toàn out-of-distribution. Hai phát hiện.

Thứ nhất, **mô hình SFT tổng quát hóa tốt nhất**: baseline SFT 1.7B đạt Overall cao nhất 0.6395 và MCQ 0.740 — cho thấy chưng cất có thể **chuyên biệt hóa quá mức** vào phân phối huấn luyện.

Thứ hai, **trong các mô hình đã chưng cất, On-Policy thắng ở suy luận**: On-Policy 1.7B đạt 0.6155, với **NLI cao nhất 0.920 và Syllogism cao nhất 0.372** toàn bảng. Điều này khẳng định: **sửa exposure bias giúp tổng quát hóa tốt hơn** trên phân phối lạ. Còn DPO trên 0.6B thì sụp xuống 0.16 — lại là vấn đề dung lượng."

*→ chuyển ý:* "Để giữ minh bạch khoa học, em xin nêu rõ các giới hạn."

---

### 🟦 Slide 13 — Limitations *(~45s)*
🎤 "Em chủ động nêu sáu giới hạn. **Bất ổn dung lượng ở 0.6B** — on-policy và DPO phản tác dụng. **Chuyên biệt hóa quá mức** trên OOD, rõ nhất ở MCQ. **Metric proxy và tổng hợp** — Syllogism dùng ROUGE-L thay vì điểm trọng tài chính thức, và Overall trộn các thang đo. **Đánh giá một lần chạy** — không có nhiều seed hay khoảng tin cậy, nên các chênh lệch nhỏ chỉ nên đọc là xu hướng. **Rò rỉ đáp án từ Oracle** ở Pha 2 — chưa ablation được. Và **phụ thuộc API ngoài cùng phần cứng giới hạn**.

Em tin việc tự nêu trước những điểm này thể hiện nghiên cứu được làm trung thực."

*→ chuyển ý:* "Tổng kết lại."

---

### 🟦 Slide 14 — Conclusion & Contributions *(~45s)*
🎤 "Đề tài có bốn đóng góp. Một — **khung chưng cất đa pha** với topology phân nhánh đã được kiểm chứng. Hai — **lượng tử hóa là thiết yếu ở quy mô nhỏ**: QLoRA như bộ điều chuẩn ẩn, tăng Syllogism khoảng 33%. Ba — **exposure bias có ý nghĩa**: on-policy nâng NLI và thắng tổng quát hóa OOD. Bốn — **chẩn đoán ngữ nghĩa cho DPO** vượt xa lọc bằng luật.

Thông điệp cuối: **một pipeline được dàn dựng cẩn thận, neo bởi lượng tử hóa, trao cho mô hình 1.7B khả năng suy luận pháp lý mạnh in-domain** — đồng thời phơi bày đánh đổi giữa chuyên biệt và tổng quát mà tương lai cần giải."

*→ chuyển ý:* "Em xin kết thúc phần trình bày."

---

### 🟦 Slide 15 — Thank You *(~10s)*
🎤 "Em xin chân thành cảm ơn quý thầy cô đã lắng nghe. Em rất mong nhận được câu hỏi và góp ý từ hội đồng ạ."

---

## PHẦN 2 — SLIDE BACKUP (Slide 16–23, chỉ lật khi bị hỏi)

> Khi hội đồng hỏi đúng chủ đề, nói *"Dạ câu này em có chuẩn bị, em xin phép lật phụ lục"* rồi nhảy tới slide tương ứng.

### Slide 16 — Anticipated Questions (1/2)
🎤 *Lật khi bị hỏi:* DPO Overall thấp hơn offline / rò rỉ đáp án Pha 2 / vì sao 0.6B sụp đổ.
"Dạ ba câu này em đã chuẩn bị. Overall là trung bình thô trộn thang đo nên không phải tiêu chí chính; DPO thắng ở từng tác vụ suy luận. Việc đưa gold cho thầy em ghi rõ ở Limitations, có hai lớp giảm thiểu. Còn 0.6B sụp đổ là do thiếu dung lượng tối thiểu cho căn chỉnh."

### Slide 17 — Anticipated Questions (2/2)
🎤 *Lật khi bị hỏi:* dùng GPT-4o-mini / sao chọn DPO thay vì PPO / các điểm bẫy.
"Dạ bảo mật là phạm vi suy luận — mô hình triển khai chạy tại chỗ, API chỉ chạm vào dựng dữ liệu một lần. DPO không cần reward model hay critic trong VRAM, hợp một GPU. Phần dưới là các phản hồi nhanh cho từng điểm bẫy."

### Slide 18 — Data: Dataset & HP Tuning (Table 4.1, 4.2)
🎤 *Khi hỏi phân bố dữ liệu / vì sao tinh chỉnh siêu tham số.*
"Dạ bảng trái là phân bố ba tác vụ qua các split; bảng phải cho thấy V1 gây quên thảm họa và code-switching, V2 ổn định lại — cấu hình V2 đầy đủ ở slide cuối phụ lục."

### Slide 19 — Data: Performance Progression (Table 4.3)
🎤 *Khi hỏi số liệu chi tiết từng pha.*
"Dạ đây là toàn bộ MCQ, NLI, Syllogism, Overall qua bốn pha cho cả 0.6B và 1.7B — thấy rõ 1.7B DPO tối đa hai tác vụ suy luận còn 0.6B sụp đổ."

### Slide 20 — Data: OOD Generalization (Table 4.4)
🎤 *Khi hỏi chi tiết VLSP2025.*
"Dạ đây là bảng đầy đủ trên 440 mẫu OOD — SFT 1.7B cao nhất Overall và MCQ, On-Policy 1.7B cao nhất NLI và Syllogism trong nhóm chưng cất."

### Slide 21 — Data: Ablations & Qualitative (Table 4.5, 4.6, 4.7)
🎤 *Khi hỏi số liệu ablation hoặc ví dụ định tính.*
"Dạ ba bảng: LoRA-vs-QLoRA, chất lượng chẩn đoán luật-vs-Judge với số tuyệt đối, và đối chiếu chín mẫu — 0.6B trượt toàn bộ Syllogism, đúng trần quy mô."

### Slide 22 — Data: Key Hyperparameters
🎤 *Khi hỏi cấu hình huấn luyện / khả năng tái lập.*
"Dạ toàn bộ siêu tham số gom ở đây: Offline KD V2, QLoRA, On-Policy, Diagnosis-DPO, mô hình và đánh giá — đủ để tái lập."

---

## 🎯 Mẹo trình bày
1. **Tốc độ:** ~130 từ/phút. Slide 10 (tiến trình các pha) nói chậm, dừng 2 giây sau câu "0.6B thì sụp đổ".
2. **Eye contact:** nhìn hội đồng ở 3 câu mở mỗi slide, chỉ liếc slide khi chỉ vào số/đồ thị.
3. **Câu cứu nguy** nếu quên: *"Dạ, ý chính của slide này là..."* rồi đọc tiêu đề slide.
4. **Khi bị hỏi khó:** thừa nhận giới hạn trước, rồi lật slide backup tương ứng — đừng phòng thủ.
5. **Tổng thời gian:** 15 slide chính ≈ 9–10 phút. Nếu bị nhắc giờ, lướt nhanh slide 6 và 11.
6. **Phân biệt cốt lõi cần thuộc:** Pipeline **phân nhánh** (P2/P3 song song từ P1), tham chiếu DPO = **checkpoint P1**, Overall là **chỉ số thô**, per-task mới chính.
