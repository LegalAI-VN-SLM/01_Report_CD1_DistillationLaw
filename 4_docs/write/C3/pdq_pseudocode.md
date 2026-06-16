# LaTeX-style Pseudocode for PDQ (Pruning, Distillation, Quantization) Pipeline

Tài liệu này trình bày giải thuật chi tiết của 3 giai đoạn trong quy trình tối ưu hóa mô hình ngôn ngữ lớn (SLM) dưới dạng mã giả học thuật (LaTeX algorithm style) thường thấy trong các bài báo nghiên cứu khoa học.

---

## Phase 1: Pruning (Wanda - Pruning LLMs by Weights and Activations)

Thuật toán **Wanda** thực hiện cắt tỉa trọng số dựa trên tích số của độ lớn trọng số (weight magnitude) và chuẩn L2 của các kích hoạt đầu vào (input activation norm). Đây là phương pháp cắt tỉa không cần huấn luyện lại (training-free).

### Algorithm 1: Wanda Pruning for LLM Layers
```latex
================================================================================
Algorithm 1: Wanda Layer-wise Pruning
================================================================================
Input: 
  - Weight matrix W \in \mathbb{R}^{d_{out} \times d_{in}} of a linear layer
  - Calibration dataset D_{calib}
  - Target sparsity level s \in [0, 1)
  
Output:
  - Pruned weight matrix W_{pruned} \in \mathbb{R}^{d_{out} \times d_{in}}
  - Binary pruning mask M \in \{0, 1\}^{d_{out} \times d_{in}}
================================================================================

1:  Initialize activation matrix X \leftarrow []
2:  For each batch b in D_{calib} do:
3:      Pass b through the network up to the current layer
4:      Collect input activations X_{b} \in \mathbb{R}^{N_b \times d_{in}} for this layer
5:      Append X_{b} to X
6:  End For
7:  Concatenate collected activations: X \in \mathbb{R}^{N \times d_{in}} where N = \sum N_b

8:  Compute column-wise L2-norm of activations:
        X_{norm, j} = \sqrt{\sum_{k=1}^N X_{k, j}^2}   for j = 1, 2, ..., d_{in}
        
9:  Initialize Mask matrix M \in \{0, 1\}^{d_{out} \times d_{in}} with 1s

10: For each row i = 1, 2, ..., d_{out} do:
11:     Compute importance scores for all input channels j:
            S_{i, j} = |W_{i, j}| \cdot X_{norm, j}
            
12:     Determine threshold score \theta_i as the s-percentile of S_{i, :}
13:     For each column j = 1, 2, ..., d_{in} do:
14:         If S_{i, j} < \theta_i then:
15:             M_{i, j} \leftarrow 0
16:         End If
17:     End For
18: End For

19: Compute pruned weight: W_{pruned} \leftarrow W \odot M
20: Return W_{pruned}, M
================================================================================
```

### PyTorch Mapping & Implementation Notes:
* **Activation norm:** $X_{norm}$ tương ứng với việc tính toán `torch.norm(X, p=2, dim=0)`.
* **Sparsity mask:** Thao tác tìm `s-percentile` tương ứng với `torch.topk` với giá trị `k = int(d_in * s)` để tìm các chỉ số nhỏ nhất cần prune (set mask = 0).
* Trong thực tế, Wanda thực hiện prune trên cấu trúc nhóm (n:m sparsity) hoặc unstructured (như mô tả ở trên).

---

## Phase 2: Knowledge Distillation (Offline Logit KD with LoRA)

Giai đoạn chưng cất tri thức sử dụng phân phối xác suất (Logits) từ mô hình Teacher lớn để định hướng mô hình Student nhỏ, kết hợp hiệu chỉnh tham số hiệu quả bằng LoRA.

### Algorithm 2: Offline Logit Distillation with LoRA
```latex
================================================================================
Algorithm 2: Offline Logit KD with LoRA Optimization
================================================================================
Input:
  - Student Model M_S parameterized by frozen weights \theta_S and trainable LoRA parameters \Phi_S
  - Dataset D containing training samples (X, Y)
  - Pre-computed Teacher Top-K Logits cache: y^T_{top\_k} and token indices I^T_{top\_k}
  - Temperature T > 0
  - Distillation coefficient \alpha \in [0, 1]
  - Learning rate \eta, Optimizer Opt
  
Output:
  - Optimized LoRA parameters \Phi_S^*
================================================================================

1:  Initialize LoRA parameters \Phi_S (typically W_A \sim \mathcal{N}, W_B = 0)
2:  For epoch = 1 to Epochs do:
3:      For each batch B = (X, Y, y^T_{top\_k}, I^T_{top\_k}) in D do:
4:          Forward pass Student: Z^S \leftarrow M_S(X; \theta_S, \Phi_S)
            // Z^S \in \mathbb{R}^{B \times L \times V} (V is Vocabulary size)
            
5:          Compute Cross-Entropy (CE) Loss on target tokens:
                L_{CE} = -\sum_{t \in \text{target}} \log \text{Softmax}(Z^S_{t, Y_t})
                
6:          Extract Student logits corresponding to Teacher's cached top-k token IDs:
                Z^S_{sel} \leftarrow \text{Gather}(Z^S, dim=-1, index=I^T_{top\_k})
                
7:          Compute Soft Target probabilities (Teacher) and Log probabilities (Student):
                P^T = \text{Softmax}(y^T_{top\_k} / T)
                \log P^S = \text{LogSoftmax}(Z^S_{sel} / T)
                
8:          Compute Kullback-Leibler (KL) Divergence loss:
                L_{KD} = T^2 \cdot \text{KL}(\log P^S \parallel P^T)
                
9:          Compute total loss:
                L = \alpha \cdot L_{KD} + (1 - \alpha) \cdot L_{CE}
                
10:         Backward pass: Compute gradients \nabla_{\Phi_S} L
11:         Update parameters: \Phi_S \leftarrow \Phi_S - \eta \cdot Opt(\nabla_{\Phi_S} L)
12:     End For
13: End For
14: Return \Phi_S^*
================================================================================
```

### PyTorch Mapping & Implementation Notes:
* **Gather logits:** `student_selected = student_logits.gather(-1, teacher_ids)` lấy ra đúng các logit của Student tại vị trí các token ID mà Teacher dự đoán cao nhất (Top-K) để so khớp phân phối.
* **KL Divergence:** Sử dụng `F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T**2)`. Nhân thêm $T^2$ giúp bảo toàn độ lớn gradient khi chia tỷ lệ logits cho $T$.

---

## Phase 3: Quantization (AWQ - Activation-aware Weight Quantization)

AWQ tối ưu hóa việc lượng tử hóa bằng cách bảo vệ các kênh trọng số quan trọng (salient channels) dựa trên độ lớn kích hoạt của dữ liệu thực tế.

### Algorithm 3: Activation-aware Weight Quantization (AWQ)
```latex
================================================================================
Algorithm 3: Activation-aware Weight Quantization (AWQ)
================================================================================
Input:
  - FP16 Weight matrix W \in \mathbb{R}^{d_{out} \times d_{in}} of a linear layer
  - Calibration activations X \in \mathbb{R}^{N \times d_{in}}
  - Grid search range for hyperparameter \beta \in [0, 1] (step size 0.1)
  
Output:
  - Quantized integer weights W_q
  - Scaling factors \mathbf{s} \in \mathbb{R}^{d_{in}}
  - Step size \Delta and Zero-point z for uniform quantization
================================================================================

1:  Compute average activation scale per input channel:
        s_X = \frac{1}{N} \sum_{i=1}^N |X_i|  \in \mathbb{R}^{d_{in}}
        
2:  Initialize best reconstruction loss L_{best} \leftarrow \infty
3:  Initialize best scaling factor \mathbf{s} \leftarrow \mathbf{1}

4:  // Grid Search for optimal scaling
5:  For \beta in [0.0, 0.1, ..., 1.0] do:
6:      Compute candidate channel scaling factor:
            s_{cand} = (s_X)^\beta
            
7:      Scale the weights and activations:
            W_{cand} = W \cdot \text{diag}(s_{cand})^{-1}
            X_{cand} = X \cdot \text{diag}(s_{cand})
            
8:      Compute quantization parameters (scale \Delta, zero z) for W_{cand}:
            \Delta = \frac{\max(W_{cand}) - \min(W_{cand})}{2^b - 1}
            z = \text{round}\left( \frac{-\min(W_{cand})}{\Delta} \right)
            
9:      Quantize and dequantize candidate weights:
            \hat{W}_{cand} = \text{clip}\left( \left\lfloor \frac{W_{cand}}{\Delta} \right\rceil + z, 0, 2^b - 1 \right)
            \tilde{W}_{cand} = (\hat{W}_{cand} - z) \cdot \Delta
            
10:     Compute output reconstruction loss (MSE):
            L = \| W \cdot X^T - (\tilde{W}_{cand} \cdot \text{diag}(s_{cand})) \cdot X^T \|_2^2
            
11:     If L < L_{best} then:
12:         L_{best} \leftarrow L
13:         \mathbf{s} \leftarrow s_{cand}
14:     End If
15: End For

16: // Apply optimal scaling
17: W' \leftarrow W \cdot \text{diag}(\mathbf{s})^{-1}
18: Compute final quantization parameters \Delta, z for W'
19: Quantize weights:
        W_q = \text{clip}\left( \left\lfloor \frac{W'}{\Delta} \right\rceil + z, 0, 2^b - 1 \right)

20: Return W_q, \mathbf{s}, \Delta, z
================================================================================
```

### PyTorch Mapping & Implementation Notes:
* **Quantization formula:** $\text{clip}(\lfloor \cdot \rceil)$ tương đương với `torch.clamp(torch.round(W / scale) + zero_point, qmin, qmax)`.
* **Activation awareness:** AWQ không lượng tử hóa trực tiếp kích hoạt $X$. Nó chỉ nhân kích hoạt với $\mathbf{s}$ để triệt tiêu việc chia $\mathbf{s}$ ở phần trọng số $W$. Khi chạy inference, phần cứng sẽ thực hiện phép nhân $Y = \tilde{W}_{cand} \cdot X = (\hat{W}_{cand} \cdot \text{diag}(\mathbf{s})) \cdot X = \hat{W}_{cand} \cdot (\text{diag}(\mathbf{s}) \cdot X)$, giúp giữ nguyên giá trị FP16 của các kênh kích hoạt quan trọng và giảm thiểu lỗi lượng tử hóa của mô hình.
