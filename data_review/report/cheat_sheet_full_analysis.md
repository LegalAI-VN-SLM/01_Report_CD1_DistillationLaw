# Cheat sheet - Full Cross-Model Analysis

## 1. Tong quan nhanh
- Bai toan: VN-Legal-AI Distillation, benchmark ViLawQA (MC 100, NLI 141, Syllogism 84)
- Mo hinh: Qwen3-0.6B (LoRA, QLoRA), Qwen3-1.7B (CPT-LoRA), Gwen3-1.7B (VLSP-PT + SFT)
- Ket luan tong: Khong co best overall. Classification thang QLoRA-0.6B. Generation thang 1.7B-CPT. Macro-F1 thang VLSP-1.7B.

## 2. Tu khoa + vi du + y nghia

### Accuracy (Acc)
- Vi du: MC Acc = 45% (QLoRA-0.6B)
- Y nghia: Ti le du doan dung tren toan bo mau.

### Null rate (Null%)
- Vi du: MC Null% = 68% (LoRA-0.6B)
- Y nghia: Ti le mau model khong tra loi hop le (format fail). Null cao => acc giam manh.

### Macro-F1
- Vi du: NLI Macro-F1 = 69.5% (VLSP-1.7B)
- Y nghia: Do can bang giua cac nhan; phat hien bias du doan lech nhan.

### Bias%
- Vi du: NLI Bias% = 97.9% (LoRA-0.6B)
- Y nghia: Ty le model nghieng ve mot nhan (relevant/not_relevant). Bias cao => classifier gan nhu vo dung.

### Confusion Matrix
- Vi du: NLI TP/FP/FN/TN thay ro model lech nhan
- Y nghia: Giai thich vi sao acc va F1 chenh nhau.

### Perplexity (PPL)
- Vi du: 4B CPT PPL = 1.83 (QLoRA)
- Y nghia: Do kho cua mo hinh tren corpus CPT; size lon => PPL tot.

### Scaling law
- Vi du: 4B (1.83) < 1.7B (1.96) < 0.6B (2.19-2.24)
- Y nghia: Model lon hon hoc ngon ngu tot hon o CPT.

### LoRA vs QLoRA
- Vi du: MC Acc 10% (LoRA-0.6B) vs 45% (QLoRA-0.6B)
- Y nghia: QLoRA giam null, giam bias, tang acc ro ret tren cung base.

### Thinking vs Direct
- Vi du: NLI Direct +11% den +12% (3/4 models)
- Y nghia: Direct mode tot hon cho classification; tiet kiem token.

### ROUGE-L (Syllogism)
- Vi du: 1.7B-CPT mean 0.5952
- Y nghia: Do do giong noi dung sinh ra voi dap an; cao hon => reasoning generative tot hon.

### Hard samples
- Vi du: MC 36/100 sai ca 4 models; NLI 10/141 sai ca 4 models
- Y nghia: Tap cau hoi can review chat luong de thi hoac bo sung kien thuc.

## 3. Thong diep chot de no khi thuyet trinh
- Classification: chat luong SFT va format compliance quan trong hon size.
- NLI: giam bias quan trong hon tang size; VLSP-1.7B can bang nhat.
- Syllogism: size quyet dinh; 1.7B-CPT thang ro.
- Direct mode nen dung cho MC/NLI de vua tang acc vua giam token.
- QLoRA la giai phap chinh cho base nho.

## 4. Khuyen nghi hanh dong (noi gon)
1) Diff config LoRA vs QLoRA SFT (root cause gap 10% vs 45% MC)
2) Apply QLoRA SFT cho 1.7B (best of both worlds)
3) Disable thinking cho classification
4) Apply QLoRA SFT cho 4B (CPT da san)
5) Phan tich hard samples
