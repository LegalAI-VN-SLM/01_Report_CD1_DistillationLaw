 # Script Thuyết Trình V2 — Capstone Project 1
### "A Hybrid Knowledge Distillation and Preference Optimization Framework for Vietnamese Legal Reasoning in Small Language Models"
**Người trình bày:** Hoàng Đình Quý Vũ — 252805008 · **GVHD:** PGS. TS. Lê Anh Cường

> Cách dùng: phần **🎤 Nói** là lời thoại. Phần *→ chuyển ý* là câu bắc cầu sang slide sau (đọc liền mạch).
> **V2 cập nhật cho deck 28 slide** (`VN_Legal_AI_CD1_v2.pptx`): đã chèn **2 slide khái niệm nền (4 & 5)** sau "Problem & Motivation", và **đánh số footer lại thành `/18`**.
> **Bài chính = slide 1–19** (17 trang đánh số 01–17/18; slide 15 là backup, lướt qua). **Phụ lục = slide 20–28** (gồm 3 slide giải thuật 20–22 và **bảng kết quả chi tiết ở slide 23**), chỉ lật khi hội đồng hỏi.
> ⏱️ Thêm 2 slide → tổng ~12 phút. Muốn giữ ~11 phút: nói 2 slide mới **gọn ~30s**, và lướt nhanh các slide công thức (9, 11, 13).

---

## PHẦN 1 — BÀI NÓI CHÍNH (Slide 1–19, ~11–12 phút)

### 🟦 Slide 1 — Title *(~20s)*
🎤 "Kính chào quý thầy cô trong hội đồng. Em là Hoàng Đình Quý Vũ, sinh viên Khoa Công nghệ Thông tin, Trường Đại học Tôn Đức Thắng. Hôm nay em xin trình bày đồ án Capstone Project 1 với đề tài *Khung tích hợp Chưng cất tri thức và Tối ưu hóa sở thích cho tác vụ Suy luận pháp lý Việt Nam trên các Mô hình ngôn ngữ nhỏ*, dưới sự hướng dẫn của thầy Lê Anh Cường."

*→ chuyển ý:* "Trước khi đi vào chi tiết, em xin tóm tắt nhanh nội dung."

---

### 🟦 Slide 2 — Table of Contents *(~20s)*
🎤 "Bài của em gồm bốn mạch: **một** — đặt vấn đề và mục tiêu; **hai** — kiến trúc pipeline và ba pha chưng cất; **ba** — thiết lập thí nghiệm; và **bốn** — kết quả, giới hạn và kết luận. Em sẽ dừng lâu nhất ở phần kết quả."

*→ chuyển ý:* "Bắt đầu từ câu hỏi: vì sao bài toán này quan trọng?"

---

### 🟦 Slide 3 — Problem & Motivation *(~55s)*
🎤 "Các mô hình ngôn ngữ nhỏ, dưới 2 tỷ tham số, rất hấp dẫn cho lĩnh vực pháp lý: độ trễ thấp, chi phí thấp, và quan trọng nhất là **chạy hoàn toàn tại chỗ** — hồ sơ vụ án không rời khỏi tổ chức.

Nhưng mô hình nhỏ thì **khó suy luận**. Có hai khó khăn: thứ nhất là **khoảng cách dung lượng** — một mô hình 0.6 đến 1.7 tỷ tham số không thể sánh với mô hình thầy lớn đã được căn chỉnh; thứ hai là **thiên lệch phơi nhiễm** (exposure bias) — lỗi sớm trong chuỗi suy luận sẽ tích lũy dọc theo chính chuỗi sinh của học sinh.

Hướng tiếp cận của em: **chưng cất khả năng suy luận của một mô hình thầy 4 tỷ tham số đã căn chỉnh bằng GRPO, vào một mô hình học sinh nhỏ gọn, có thể triển khai tại chỗ**."

*→ chuyển ý:* "Nhưng trước khi nói em làm thế nào, em xin dừng một nhịp để giải thích khái niệm cốt lõi: chưng cất tri thức thực chất là gì."

---

### 🟩 Slide 4 — Background · What is Knowledge Distillation? *(MỚI · ~40s)*
🎤 "Chưng cất tri thức, ý tưởng rất đơn giản: một mô hình **thầy** lớn và giỏi truyền lại tri thức cho một mô hình **trò** nhỏ gọn.

Điểm mấu chốt nằm ở **cách** truyền. Trò không chỉ học **đáp án cuối** — cái mà ta gọi là *hard label*, ví dụ nhãn 'relevant'. Trò học **toàn bộ phân phối xác suất mềm** của thầy — *soft label*: tức là thầy **tự tin đến mức nào** giữa các lựa chọn. Đây gọi là 'dark knowledge'. Nhờ đó trò học được **cách suy luận**, chứ không chỉ kết quả.

Liên hệ ngành luật cho dễ hình dung: giống như học **quá trình lập luận của một luật sư kỳ cựu**, chứ không phải học thuộc lòng phán quyết của họ — vì trong luật, chỉ **một suy luận sai có thể làm mất hiệu lực cả bản án**."

*→ chuyển ý:* "Trong đề tài, khái niệm này được hiện thực hóa bằng ba chiến lược chưng cất khác nhau — em xin lướt qua bằng ngôn ngữ đời thường."

---

### 🟩 Slide 5 — Three Distillation Strategies *(MỚI · ~40s)*
🎤 "Pipeline của em không phải một kỹ thuật đơn lẻ, mà là **ba chiến lược**, mỗi cái vá một điểm yếu khác nhau.

**Một — Offline Logit KD, Pha 1**: trò sao chép câu trả lời mềm của thầy trên một tập dữ liệu cố định. Đây là **nền móng** kiến thức; QLoRA giữ cho trò không quên.

**Hai — On-Policy KD, Pha 2**: trò học từ **chính câu nó tự sinh ra**; thầy — được phép nhìn đáp án vàng — sửa lỗi. Cái này vá **exposure bias**, tức lỗi tích lũy khi mô hình tự sinh.

**Ba — Diagnosis-Driven DPO, Pha 3**: một **LLM đóng vai trọng tài** chẩn đoán lỗi suy luận của trò rồi biến thành cặp sở thích; DPO đẩy mô hình **tránh xa logic sai và 'đoán may'**.

Ba chiến lược này ánh xạ đúng ba khoảng trống nghiên cứu ngay slide sau."

*→ chuyển ý:* "Vậy ba khoảng trống đó cụ thể là gì?"

---

### 🟦 Slide 6 — Research Gap & Objectives *(~50s)*
🎤 "Vấn đề đã rõ. Nhưng vì sao đến giờ vẫn chưa có lời giải trọn vẹn cho tiếng Việt? Em thấy **ba khoảng trống**.

**Thứ nhất**, chưng cất kiểu cũ chỉ dạy mô hình trên câu mẫu cố định — nó **bỏ qua exposure bias**, nên suy luận nhiều bước vẫn dễ **sai dây chuyền**.

**Thứ hai**, ở khâu căn chỉnh, người ta lọc dữ liệu bằng **luật cứng**. Mà luật cứng thì dễ vỡ: nó **khen nhầm** câu 'đoán may', và **loại oan** câu đúng nhưng khác định dạng.

**Thứ ba**, gần như **chưa ai thử** những kỹ thuật này ở quy mô **dưới hai tỷ tham số**, cho tiếng Việt.

Ba khoảng trống đó **định hình đúng những gì em làm**: on-policy KD để vá exposure bias, DPO dẫn dắt bằng chẩn đoán để thay luật cứng, và kiểm chứng trên cả **dữ liệu quen và dữ liệu lạ** (VLSP2025) — tất cả gói trong **một khung lai duy nhất**, neo bằng QLoRA."

*→ chuyển ý:* "Năm mục tiêu này được hiện thực hóa trong một pipeline thống nhất."

---

### 🟦 Slide 7 — Proposed Pipeline *(~60s)*
🎤 "Đây là bức tranh tổng thể, và em xin nhấn mạnh một điểm hay bị hiểu nhầm: **đây không phải chuỗi tuyến tính ba pha nối tiếp**.

Gốc chung duy nhất là **mô hình SFT**. Từ SFT, em rẽ thành **hai nhánh độc lập, song song**: nhánh trên là **On-Policy KD (Pha 2)** — đi thẳng từ SFT — ra mô hình On-Policy SLM; nhánh dưới là **nhánh offline: Pha 1 (chưng cất logit offline) rồi nối tiếp Pha 3 (DPO dẫn dắt bằng chẩn đoán)** ra mô hình DPO-Aligned SLM.

Em xin nhấn mạnh để tránh hiểu nhầm: **Pha 2 khởi tạo từ SFT, KHÔNG phải từ Pha 1**; còn **Pha 3 thì nối tiếp từ checkpoint Pha 1**, và chính checkpoint Pha 1 đó cũng là **chính sách tham chiếu của DPO**. Mô hình thầy là **Qwen3-4B Vietnamese-legal-GRPO**."

*→ chuyển ý:* "Ta đi vào từng pha. Mỗi pha em trình bày sơ đồ trước, công thức sau. Đầu tiên là Pha 1."

---

### 🟦 Slide 8 — Pha 1 · Offline Logit KD (SƠ ĐỒ) *(~40s)*
🎤 "Pha 1 lo phần **thu nhận kiến thức nền**. Học sinh **bắt chước phân phối mềm** của thầy trên một tập dữ liệu tĩnh.

Quý thầy cô nhìn sơ đồ: cùng một input đưa vào **cả thầy và trò**. Thầy đóng băng, chỉ lấy **Top-50 logit**; trò gắn **adapter QLoRA**. Hai phía sinh ra hai hàm loss — một loss mềm so với thầy, một loss cứng so với đáp án vàng — rồi tổ hợp lại để cập nhật trò.

Điểm cốt lõi, và là một phát hiện của em: **QLoRA đóng băng base ở 4-bit và chỉ cập nhật adapter — hoạt động như một bộ điều chuẩn ẩn, ngăn quên thảm họa** ở quy mô nhỏ."

*→ chuyển ý:* "Cụ thể hai hàm loss đó viết ra sao? Đây là công thức."

---

### 🟦 Slide 9 — Pha 1 · Công thức *(~40s)*
🎤 "Tổng mất mát là **alpha nhân Soft Loss cộng một-trừ-alpha nhân Hard Loss**, với alpha bằng 0.3 — tức là **70% trọng số đặt vào neo cross-entropy**, 30% vào chưng cất.

**Soft Loss** là KL-divergence giữa phân phối thầy và trò; cả hai được chia cho **nhiệt độ T** rồi chuẩn hóa **chỉ trên Top-50 token** lớn nhất của thầy, và nhân thêm **T-bình-phương** để giữ độ lớn gradient.

**Hard Loss** là cross-entropy trên token vàng, chỉ tính ở phần câu trả lời — token của prompt được mask để không tính loss. Chính cái neo CE này giữ cho trò không chạy theo nhiễu của thầy."

*→ chuyển ý:* "Nhưng offline có điểm yếu cố hữu: nó dạy trên câu vàng, không phải trên chuỗi trò tự sinh — đó là exposure bias. Pha 2 giải quyết việc này."

---

### 🟦 Slide 10 — Pha 2 · On-Policy KD (SƠ ĐỒ) *(~45s)*
🎤 "Pha 2 giải quyết exposure bias. Thay vì học trên câu vàng cố định, học sinh **sinh ra chính rollout của nó** với decoding ngẫu nhiên — nhiệt độ 0.7, top-p 0.9.

Theo sơ đồ: rollout đó được đưa lên cho **thầy ở chế độ Oracle** — thầy có **quyền truy cập đặc biệt vào đáp án vàng y-sao**. Thầy chấm sửa lỗi ở **cấp token**, rồi tín hiệu quay xuống cập nhật học sinh qua **KL có cắt ngưỡng**.

Ý nghĩa: huấn luyện trên chính quỹ đạo tự sinh **khép lại khoảng cách giữa huấn luyện và suy luận**. Em xin thẳng thắn — việc đưa đáp án vàng cho thầy là một **đường rò rỉ tiềm tàng**, em có ghi rõ trong Giới hạn."

*→ chuyển ý:* "Công thức của pha này, đặc biệt là cơ chế cắt KL, như sau."

---

### 🟦 Slide 11 — Pha 2 · Công thức *(~40s)*
🎤 "Tổng loss cũng gồm hai phần, lần này alpha bằng 0.5 — cân bằng giữa loss on-policy và neo SFT.

KL ở đây tính **từng token trên toàn bộ từ vựng**, trên chuỗi do **trò tự sinh**. Điểm đặc biệt là **KL Clipping**: mỗi giá trị KL bị cắt tại ngưỡng **C bằng 5** — vì các token cấu trúc như think hay major_premise có KL khổng lồ, nếu không cắt sẽ **lấn át gradient** của nội dung luật pháp.

Loss on-policy cuối cùng là **trung bình các KL đã cắt**, nhân T-bình-phương."

*→ chuyển ý:* "Hai pha trên transfer kiến thức. Còn nhánh offline cần một bước nữa để **sửa lỗi suy luận còn sót** — đó là Pha 3, nối tiếp từ checkpoint Pha 1."

---

### 🟦 Slide 12 — Pha 3 · Diagnosis-Driven DPO (SƠ ĐỒ) *(~45s)*
🎤 "Pha 3 căn chỉnh để sửa các lỗi suy luận còn sót. Theo sơ đồ: học sinh sinh rollout, rồi một **LLM Judge — GPT-4o-mini — kiểm toán ngữ nghĩa** cả chuỗi suy nghĩ lẫn kết luận cuối.

Nếu **không có lỗi (OK)** thì bỏ. Nếu **có lỗi**, mẫu được đưa qua **lấy mẫu phân tầng** tạo thành **cặp sở thích**: chọn là **đáp án vàng**, loại là **chính lỗi mà học sinh mắc**. Cặp này đi vào **DPO**, với chính sách tham chiếu là **checkpoint Pha 1**.

Điểm mấu chốt: **chẩn đoán bằng ngữ nghĩa cho tập sở thích sạch hơn hẳn lọc bằng luật** — có số liệu ablation ở phụ lục."

*→ chuyển ý:* "Hàm DPO viết ra như sau."

---

### 🟦 Slide 13 — Pha 3 · Công thức *(~40s)*
🎤 "Đây là hàm mất mát DPO. Với mỗi bộ ba — câu hỏi x, đáp án vàng y-w (chosen), và câu lỗi y-l (rejected) — ta tối thiểu hóa **trừ log-sigmoid của hiệu hai phần thưởng ngầm**.

Phần thưởng ngầm của một câu là **beta nhân log tỉ số** giữa xác suất của chính sách đang học và chính sách tham chiếu. Ta lấy phần thưởng của câu chosen **trừ** phần thưởng của câu rejected.

Beta bằng 0.1; tham chiếu là checkpoint Pha 1. Trực giác đơn giản: **kéo xác suất câu đúng lên, đẩy câu sai xuống**, nhưng beta giữ cho mô hình **không trôi quá xa** khỏi điểm xuất phát."

*→ chuyển ý:* "Đó là toàn bộ phương pháp. Giờ sang thiết lập thí nghiệm."

---

### 🟦 Slide 14 — Experimental Setup & Data *(~45s)*
🎤 "Dữ liệu dựa trên bộ **ViLawQA** từ tài khoản thangvip, gồm ba tác vụ: trắc nghiệm MCQ, suy luận NLI ba lớp, và tam đoạn luận Syllogism. Tập huấn luyện 2.603 mẫu, validation 322, test nội bộ 325 — đây là phần in-distribution. Ngoài ra em dùng **VLSP2025 Public-Test 440 mẫu làm benchmark out-of-distribution**.

Thầy là Qwen3-4B legal-GRPO; trò là Qwen3 0.6B và 1.7B. Em lưu ý: điểm **Overall chỉ là trung bình không trọng số** trộn accuracy với ROUGE-L, nên nó là chỉ số tóm tắt thô — **điểm theo từng tác vụ mới là cơ sở chính** để so sánh."

*→ chuyển ý:* "Và bây giờ là kết quả — phần em muốn hội đồng lưu ý nhất."

---

### ⬜ Slide 15 — Experimental Setup (Backup) *(LƯỚT QUA · ~5s)*
🎤 *(Slide backup siêu tham số, không nằm trong luồng 10 phút — lướt nhanh.)* "Toàn bộ siêu tham số chi tiết em để ở slide này và phụ lục, sẵn sàng khi hội đồng cần."

*→ chuyển ý:* "Đi thẳng vào kết quả."

---

### 🟦 Slide 16 — Results *(~75s — DỪNG LÂU)*
🎤 "Đây là phần kết quả — em xin nhấn mạnh ngay: **điểm theo từng tác vụ mới là cơ sở chính**, Overall chỉ tham khảo.

**In-distribution**: mô hình 1.7B sau **DPO** đạt **NLI Accuracy cao nhất 0.8156** và **Syllogism ROUGE-L cao nhất 0.5684** — đúng hai tác vụ suy luận cốt lõi, vượt xa SFT (0.745 và 0.207). Về Overall, bản **Offline KD 1.7B cao nhất với 0.5322**. Chỉ riêng MCQ là hơi thấp hơn baseline.

Ngược lại, mô hình **0.6B sụp đổ** khi căn chỉnh: Overall rơi từ 0.4622 xuống 0.3104 rồi 0.1861 — thiếu dung lượng tối thiểu.

**Out-of-distribution trên VLSP2025**: **SFT 1.7B tổng quát hóa tốt nhất** (Overall 0.6395, MCQ 0.740); nhưng trong nhóm chưng cất, **On-Policy 1.7B dẫn đầu** với **NLI Accuracy 0.920** và Syllogism ROUGE-L 0.372.

Thông điệp một câu: **chưng cất cho ra SLM 1.7B mạnh nhất ở suy luận trong miền; SFT giữ lợi thế ở MCQ và dữ liệu lạ** — một đánh đổi giữa chuyên biệt và tổng quát. Bảng số liệu đầy đủ em để ở phụ lục (slide 23)."

*→ chuyển ý:* "Để giữ minh bạch khoa học, em xin nêu rõ các giới hạn."

---

### 🟦 Slide 17 — Limitations *(~45s)*
🎤 "Em chủ động nêu sáu giới hạn. **Bất ổn dung lượng ở 0.6B** — on-policy và DPO phản tác dụng. **Chuyên biệt hóa quá mức** trên OOD, rõ nhất ở MCQ. **Metric proxy và tổng hợp** — Syllogism dùng ROUGE-L thay vì điểm trọng tài chính thức, và Overall trộn các thang đo. **Đánh giá một lần chạy** — không có nhiều seed hay khoảng tin cậy, nên các chênh lệch nhỏ chỉ nên đọc là xu hướng. **Rò rỉ đáp án từ Oracle** ở Pha 2 — chưa ablation được. Và **phụ thuộc API ngoài cùng phần cứng giới hạn**.

Em tin việc tự nêu trước những điểm này thể hiện nghiên cứu được làm trung thực."

*→ chuyển ý:* "Tổng kết lại."

---

### 🟦 Slide 18 — Conclusion & Contributions *(~45s)*
🎤 "Đề tài có bốn đóng góp. Một — **khung lai kết hợp chưng cất tri thức và tối ưu hóa sở thích**, với topology phân nhánh đã được kiểm chứng. Hai — **lượng tử hóa là thiết yếu ở quy mô nhỏ**: QLoRA như bộ điều chuẩn ẩn, tăng Syllogism khoảng 33%. Ba — **exposure bias có ý nghĩa**: on-policy nâng NLI và thắng tổng quát hóa OOD. Bốn — **chẩn đoán ngữ nghĩa cho DPO** vượt xa lọc bằng luật.

Thông điệp cuối: **một pipeline được dàn dựng cẩn thận, neo bởi lượng tử hóa, trao cho mô hình 1.7B khả năng suy luận pháp lý mạnh in-domain** — đồng thời phơi bày đánh đổi giữa chuyên biệt và tổng quát mà tương lai cần giải."

*→ chuyển ý:* "Em xin kết thúc phần trình bày."

---

### 🟦 Slide 19 — Thank You *(~10s)*
🎤 "Em xin chân thành cảm ơn quý thầy cô đã lắng nghe. Em rất mong nhận được câu hỏi và góp ý từ hội đồng ạ."

---

## PHẦN 2 — PHỤ LỤC (Slide 20–28, chỉ lật khi bị hỏi)

> Khi hội đồng hỏi đúng chủ đề, nói *"Dạ câu này em có chuẩn bị, em xin phép lật phụ lục"* rồi nhảy tới slide tương ứng.

### Slide 20 — Algorithm 1: Phase 1 Offline Logit KD
🎤 *Khi hỏi chi tiết giải thuật Pha 1.*
"Dạ đây là mã giả Pha 1. Logit Top-50 của thầy được **tính trước và lưu cache** — nên vòng lặp huấn luyện không cần chạy thầy, đó là ý nghĩa của 'offline'. Với mỗi minibatch: lấy logit thầy từ cache; cho trò forward **teacher-forced** trên đáp án vàng; chuẩn hóa cả hai phân phối với nhiệt độ T; tính **KD loss** và **CE loss**; tổ hợp theo alpha; rồi cập nhật **chỉ adapter LoRA**. Kết thúc trả về **checkpoint Pha 1** — chính là gốc cho Pha 3."

### Slide 21 — Algorithm 2: Phase 2 On-Policy KD
🎤 *Khi hỏi chi tiết giải thuật Pha 2.*
"Dạ mã giả Pha 2. Với mỗi mẫu, **trò tự sinh rollout** với nhiệt độ 0.7, top-p 0.9. Sau đó lặp qua **từng token** của rollout: thầy ở chế độ **Oracle** — điều kiện trên rollout cộng đáp án vàng — cho ra phân phối; trò cho ra phân phối của nó; tính KL rồi **cắt tại ngưỡng C bằng 5**. Loss on-policy là trung bình các KL đã cắt, cộng thêm **neo SFT** để chống quên. Trả về mô hình **On-Policy SLM**. Em lưu ý: trò khởi tạo từ **SFT**, độc lập với Pha 1."

### Slide 22 — Algorithm 3: Phase 3 Diagnosis-Driven DPO
🎤 *Khi hỏi chi tiết giải thuật Pha 3.*
"Dạ giải thuật Pha 3 gồm hai giai đoạn. **Giai đoạn A — dựng dữ liệu sở thích**: trò sinh rollout; **LLM Judge** gán nhãn OK, RISKY, PARTIAL hay WRONG; chỉ những mẫu **RISKY và WRONG** mới thành cặp — chọn là đáp án vàng, loại là chính câu lỗi của trò — rồi **lấy mẫu phân tầng** cân bằng tác vụ. **Giai đoạn B — căn chỉnh**: chạy DPO trên các cặp đó. Cả **khởi tạo lẫn tham chiếu** đều là checkpoint Pha 1. Trả về mô hình **DPO-Aligned SLM**."

### Slide 23 — Results · Final Evaluation (BẢNG CHI TIẾT SFT vs Distilled)
🎤 *Lật khi hội đồng muốn xem bảng số đầy đủ (đây là bảng chi tiết cho phần Results ở slide 16).*
"Dạ đây là bảng chi tiết: chưng cất 1.7B tốt nhất so với SFT — NLI 0.745 lên 0.816, Syllogism 0.207 lên 0.568, Overall 0.484 lên 0.532; chỉ MCQ giảm. Bên phải là out-of-distribution: SFT giữ lợi thế Overall và MCQ, On-Policy dẫn đầu NLI 0.920. Cùng thông điệp với phần kết quả: đánh đổi chuyên biệt vs tổng quát."

### Slide 24 — Data: Dataset & HP Tuning
🎤 *Khi hỏi phân bố dữ liệu / vì sao tinh chỉnh siêu tham số.*
"Dạ bảng phân bố ba tác vụ qua các split; và bảng tinh chỉnh cho thấy V1 (LR 1e-4, batch 16) gây quên thảm họa và code-switching Việt–Anh–Trung, còn V2 (LR 2e-5, batch 64, alpha 0.3) ổn định lại, CE hội tụ 0.5 → 0.25."

### Slide 25 — Data: Chi phí tính toán (Peak VRAM & Runtime)
🎤 *Khi hỏi chi phí GPU / tài nguyên từng pha huấn luyện.*
"Dạ hai biểu đồ chi phí theo pha. **Peak VRAM**: rẻ nhất là On-Policy 0.6B (11.9G) và các pha DPO (13–15G); nặng nhất là **On-Policy 1.7B 45.2G**, rồi Offline 1.7B 22.2G — vì phải giữ đồng thời thầy (Top-50 logit) và trò trong bộ nhớ. **Runtime**: DPO nhanh nhất (~0.9–1.3h), Offline ~1.3–1.4h; đắt nhất là **On-Policy** do có vòng sinh rollout mỗi bước — 1.7B mất 3.97h, riêng 0.6B lên tới 10.88h (nhiều bước, nhiễu cao). Thông điệp: on-policy đắt nhất nhưng tất cả vẫn gọn trong một GPU nhờ QLoRA."

### Slide 26 — Data: Ablation QLoRA vs LoRA (CE loss)
🎤 *Khi hỏi bằng chứng QLoRA chống quên.*
"Dạ đường CE loss: **LoRA V1** (LR 1e-4, alpha 0.5) quên thảm họa — CE cao ~1.00 → 0.56 rồi trôi ngược lên; **LoRA V2** (LR 2e-5, alpha 0.3) đỡ hơn nhưng vẫn nhiễu quanh 0.5; **QLoRA V2** giảm đều 0.64 → 0.37, thấp nhất suốt quá trình — bằng chứng QLoRA neo được kiến thức SFT; Syllogism R-L nhờ đó tăng 0.1741 → 0.3804, khoảng +33%."

### Slide 27 — Data: On-Policy Loss Curves (0.6B vs 1.7B)
🎤 *Khi hỏi động lực học huấn luyện on-policy theo quy mô.*
"Dạ đường loss on-policy: 0.6B hội tụ về ~1.07, nhiễu nhiều, plateau cao; 1.7B hội tụ mượt về ~0.73 sau cú warmup spike. Ý nghĩa: mô hình lớn hấp thụ được tín hiệu on-policy, còn 0.6B kẹt ở mức cao — 'sụp đổ' thực sự thể hiện ở điểm test."

### Slide 28 — Data: DPO Reward Margin, Pha 3 (Chosen − Rejected)
🎤 *Khi hỏi DPO có tách được chosen/rejected không, và chất lượng cặp preference.*
"Dạ biểu đồ reward margin (chosen − rejected) trên **tập train** Pha 3: **1.7B (A3)** tăng đều **−13.7 → +30** — căn chỉnh sở thích ổn định; **0.6B (A2)** cũng nới rộng **−13.4 → +22** nhưng thấp hơn, và đây chỉ là tách trên tập train nên chưa chắc tổng quát ra test. Về chất lượng cặp: **LLM Judge (GPT-4o-mini)** nâng OK 11.1%→20.7%, phát hiện WRONG 40.5%→63.5%, ép RISKY ('đoán may') 34.5%→15.5% — cặp sở thích sạch hơn nên margin mới tách tốt."

---

## 🎯 Mẹo trình bày (V2)
1. **Tốc độ:** ~130 từ/phút. Slide 16 (kết quả) nói chậm, dừng 2 giây sau câu "đánh đổi giữa chuyên biệt và tổng quát".
2. **2 slide mới (4–5):** nói **gọn, plain-language**, đừng sa đà công thức — mục đích chỉ để hội đồng ngoài NLP có điểm neo. Nếu bị nhắc giờ, gộp 4+5 thành ~40s tổng.
3. **Cặp sơ đồ + công thức (8–13):** trình bày liền mạch — sơ đồ trả lời "làm gì", công thức trả lời "bằng cách nào". Nếu thiếu giờ, **lướt nhanh slide công thức (9, 11, 13)**, giữ kỹ phần sơ đồ và kết quả.
4. **Eye contact:** nhìn hội đồng ở 3 câu mở mỗi slide, chỉ liếc slide khi chỉ vào số/công thức.
5. **Câu cứu nguy** nếu quên: *"Dạ, ý chính của slide này là..."* rồi đọc tiêu đề slide.
6. **Khi bị hỏi khó:** thừa nhận giới hạn trước, rồi lật slide phụ lục tương ứng (giải thuật 20–22, số liệu 23–28) — đừng phòng thủ.
7. **Tổng thời gian:** 19 slide chính ≈ 11–12 phút (đã cộng 2 slide mới).
8. **Phân biệt cốt lõi cần thuộc:** Pipeline **phân nhánh** (P2 từ SFT, P3 nối P1), tham chiếu DPO = **checkpoint P1**, Overall là **chỉ số thô**, per-task mới chính. Hard label = đáp án; soft label = phân phối 'dark knowledge'.

---
> ⚠️ **Ghi chú cấu trúc deck:** bảng kết quả chi tiết (slide 23) hiện nằm ở **khu phụ lục**, còn slide 16 "Results" trong luồng chính chỉ mang lời dẫn. Nếu muốn, có thể **chuyển bảng kết quả (slide 23) lên ngay sau slide 16** để phần Results trong bài chính có bảng số trực quan — báo mình nếu bạn muốn làm.
