# Script thuyet trinh (10-12 phut)

## 0. Mo dau (45-60s)
Xin chao quy thay co va cac ban. Hom nay em trinh bay bao cao tong hop “Full Cross-Model Analysis” cho bai toan VN-Legal-AI Distillation. Muc tieu la so sanh 4 mo hinh SFT tren 3 nhiem vu chinh: Multiple Choice (MC), Natural Language Inference (NLI), va Syllogism. Benchmark la ViLawQA voi 100 cau MC, 141 mau NLI, va 84 mau Syllogism.

Cac mo hinh gom:
- Qwen3-0.6B CPT + QLoRA SFT
- Qwen3-0.6B CPT + LoRA SFT
- Qwen3-1.7B CPT + LoRA SFT
- Gwen3-1.7B pretrained VLSP + SFT

Em se trinh bay 5 phan: (1) Tong quan ranking, (2) CPT perplexity, (3) MC deep dive, (4) NLI deep dive, (5) Syllogism deep dive, va ket thuc bang khuyen nghi hanh dong.

## 1. Tong quan ket qua (2 phut)
O trang Overview Dashboard, chung ta thay khong co mot mo hinh “best overall”.

- MC Accuracy: QLoRA-0.6B dan dau 45%, theo sau 1.7B-CPT 32%, VLSP-1.7B 16%, LoRA-0.6B 10%.
- NLI Accuracy: QLoRA-0.6B 71.6% cao nhat, VLSP-1.7B 69.5% sat sau, 1.7B-CPT 65.2%, LoRA-0.6B 56.7%.
- NLI Macro-F1: VLSP-1.7B vuot troi 69.5%, cac mo hinh con lai thap hon ro ret.
- Syllogism ROUGE-L: 1.7B-CPT cao nhat 0.5952, QLoRA-0.6B 0.4832, VLSP-1.7B 0.3897, LoRA-0.6B 0.3517.

Thong diep chinh: Classification thi QLoRA-0.6B tot nhat. Generation thi 1.7B-CPT tot nhat. Can bang F1 thi VLSP-1.7B tot nhat.

## 2. CPT Perplexity va scaling (1 phut)
Bang CPT Perplexity cho thay:
- 4B co PPL tot nhat: 1.83 (QLoRA) va 1.86 (LoRA).
- 1.7B PPL ~1.96.
- 0.6B PPL ~2.19-2.24.

Xu huong ro: model size lon hon thi PPL tot hon. LoRA va QLoRA gan nhu tuong duong o giai doan CPT, do cung dung mot corpus 745K tokens. Diem quan trong: 4B CPT da san sang, chua co SFT. Day la co hoi lon de apply QLoRA SFT recipe.

## 3. MC Deep Dive (2-2.5 phut)
### 3.1 Format compliance va bias
Nhan xet quan trong nhat la null rate quyet dinh accuracy.
- QLoRA-0.6B: null 19% va acc 45%.
- LoRA-0.6B: null 68% va acc chi 10%.
Hai mo hinh dung cung base, nhung QLoRA giam null manh, dan toi acc cao.

Ve position bias: LoRA-0.6B bi C-bias rat nang (75% du doan C). QLoRA-0.6B phan bo deu hon tren A/B/C/D.

### 3.2 Accuracy theo nhan
QLoRA-0.6B du doan deu tren tat ca nhan, dac biet nhan D dat 61.9%. LoRA-0.6B gan nhu khong bao gio dung nhan B hoac D.

### 3.3 Mau kho
Co 36/100 cau MC bi sai boi ca 4 mo hinh. Day la tap “hard MC” can kiem tra lai chat luong de thi hoac nhung cau can kien thuc ngoai corpus CPT.

Ket luan MC: Chat luong SFT va format compliance quan trong hon model size. QLoRA giup giam null va bias ro ret.

## 4. NLI Deep Dive (2-2.5 phut)
### 4.1 Confusion Matrix va bias
NLI cho thay bias la yeu to quyet dinh F1.
- LoRA-0.6B bias 97.9% ve “relevant”, gan nhu vo dung.
- QLoRA-0.6B bias 55.3% va acc cao nhat 71.6%.
- VLSP-1.7B bias thap nhat 41.8%, Macro-F1 cao nhat 69.5%.

Diem dang chu y: VLSP-1.7B co recall not_relevant 81% cao nhat, nhung recall relevant thap. Nghia la mo hinh nay can bang hon, do data pretraining VLSP.

### 4.2 Thinking vs Direct
Direct mode tot hon cho NLI o 3/4 mo hinh, delta tang +11% den +12%. Chi LoRA-0.6B da collapse nen ca hai mode deu kem.

### 4.3 Hard NLI
10/141 mau NLI sai boi ca 4 model, tat ca la nhan not_relevant. Can xem lai tap nay de giam bias.

Ket luan NLI: Giam bias quan trong hon tang size. Direct mode nen duoc uu tien cho NLI.

## 5. Syllogism Deep Dive (2 phut)
Syllogism la bai toan generative. Ket qua cho thay:
- 1.7B-CPT co mean ROUGE-L 0.5952 cao nhat, distribution tap trung 0.5-0.8.
- QLoRA-0.6B o muc trung binh 0.4832.
- LoRA-0.6B va VLSP-1.7B thap hon.

Thong diep ro: model size la yeu to quyet dinh cho generative reasoning. SFT trick khong the bu dap cho model qua nho.

Ve thinking vs direct, khong co pattern ro rang. Chi QLoRA-0.6B co loi the nho khi thinking. Nhung voi 1.7B, direct mode co xu huong tot hon nhe.

## 6. Tong ket va khuyen nghi (1-1.5 phut)
### 6.1 Tong ket chinh
- Khong co best overall: classification thang QLoRA-0.6B, generation thang 1.7B-CPT, F1 can bang thang VLSP-1.7B.
- Direct mode tot hon cho classification, tiet kiem ~40% tokens.
- QLoRA thang LoRA tren cung base va CPT, giam null va bias ro ret.

### 6.2 Khuyen nghi hanh dong
1) Diff LoRA vs QLoRA SFT config: tim root cause gap 10% vs 45% MC, ROI cao, khong ton compute.
2) Apply QLoRA SFT cho 1.7B: ky vong best of both worlds (MC > 50%, NLI > 75%, Syllo > 0.6).
3) Disable thinking cho classification: giam cost va tang acc.
4) Apply QLoRA SFT cho 4B: CPT PPL 1.83 da san, tiem nang cao.
5) Phan tich hard samples (MC va NLI) de nang chat luong data.

## 7. Ket luan (20-30s)
De ket lai, ket qua cho thay chat luong SFT va kiem soat bias quan trong khong kem model size. QLoRA la giai phap duoc chon cho classification, trong khi generative reasoning can model lon hon. Buoc tiep theo la chuan hoa SFT recipe va day manh 1.7B va 4B theo QLoRA.

Em xin cam on va san sang tra loi cau hoi.
