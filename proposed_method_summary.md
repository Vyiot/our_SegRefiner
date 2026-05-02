# Tài liệu Phương pháp Đề xuất: Multi-Level Mask Denoiser

Tài liệu này tổng hợp toàn bộ nội dung từ chương **III. PROPOSED METHOD** của bài báo, bao gồm kiến trúc SegRefiner gốc và các cải tiến quan trọng về mô hình nhiễu đa cấp.

---

## III. TỔNG QUAN PHƯƠNG PHÁP ĐỀ XUẤT

Chúng tôi xây dựng dựa trên khung khuếch tán rời rạc (discrete diffusion framework) của SegRefiner nhưng thiết kế lại hoàn toàn cả hai giai đoạn của quá trình forward: 
1. Tạo mặt nạ thô (**coarse mask generation**)
2. Lấy mẫu Q (**Q-sampling**)

Điểm khác biệt chính là thay thế cơ chế nhiễu chỉ ở đường biên (boundary-only noise) vốn không phụ thuộc vào ảnh của SegRefiner bằng một **mô hình nhiễu đa cấp** (multi-level noise model). Mô hình này hoạt động ở cả cấp độ đối tượng (object level) và cấp độ pixel (pixel level), được dẫn dắt bởi việc ước tính độ bất định (uncertainty estimation) từ ảnh RGB.

---

## A. Khung SegRefiner (SegRefiner Framework)

Trong quá trình huấn luyện, SegRefiner học cách khôi phục mặt nạ sạch ($M_{fine}$) từ các mặt nạ bị làm nhiễu nhân tạo. 

1. **Tạo mặt nạ coarse ($M_{coarse}$):** Làm nhiễu đường biên của mặt nạ chuẩn bằng các phép toán hình thái học (dilation/erosion) cho đến khi IoU < 0.6.
2. **Q-sampling (Huấn luyện):** Mỗi pixel giữ giá trị chuẩn với xác suất $\bar{\beta}_t$ hoặc lấy giá trị thô với xác suất $1-\bar{\beta}_t$.
3. **Mô hình:** Sử dụng Conditional U-Net nhận đầu vào là $[I || m_t]$ (ảnh RGB kết hợp với mặt nạ nhiễu).
4. **Hàm Loss:** Kết hợp giữa Binary Cross-Entropy (BCE) và Texture Loss (L1 trên gradient biên).

---

## Thuật toán (Algorithms)

### Algorithm 1: SegRefiner Training
**Require:** Dataset $\{(I_n, M_{fine}^{(n)})\}_{n=1}^N$; noise schedule $\{\bar{\beta}_t\}_{t=0}^T$; U-Net $f_\theta$  
**Ensure:** Trained parameters $\theta$

1. **repeat**
2. &nbsp;&nbsp;Lấy mẫu một cặp huấn luyện $(I, M_{fine})$ từ dataset.
3. &nbsp;&nbsp;Tạo mặt nạ thô từ ground truth ($M_{coarse}$).
4. &nbsp;&nbsp;// **Q-sampling để tạo mặt nạ nhiễu**
5. &nbsp;&nbsp;Lấy mẫu timestep $t \sim Uniform\{0, 1, \dots, T-1\}$
6. &nbsp;&nbsp;**for** mỗi pixel $(i, j)$ **do**
7. &nbsp;&nbsp;&nbsp;&nbsp;$u^{i,j} \sim Uniform(0,1)$
8. &nbsp;&nbsp;&nbsp;&nbsp;$m_t^{i,j} \gets F_{noise}(M_{fine}^{i,j}, M_{coarse}^{i,j}, \bar{\beta}_t, u^{i,j})$
9. &nbsp;&nbsp;**end for**
10. &nbsp;&nbsp;// **Forward pass và tính toán loss**
11. &nbsp;&nbsp;$z_t \gets [I || m_t]$  // Ghép ảnh và mặt nạ nhiễu
12. &nbsp;&nbsp;$logits \gets f_\theta(z_t, t)$ // Dự đoán của U-Net
13. &nbsp;&nbsp;$\mathcal{L} \gets BCE(logits, M_{fine})$
14. &nbsp;&nbsp;Cập nhật $\theta$ qua AdamW với $\mathcal{L}$
15. **until** hội tụ (120k iterations)

### Algorithm 2: SegRefiner Inference — Multi-Step Reverse Process
**Require:** Ảnh $I$; mặt nạ thô $M_{coarse}$; U-Net đã huấn luyện $f_\theta$; schedule $\{\bar{\beta}_t\}_{t=0}^T$  
**Ensure:** Mặt nạ tinh chỉnh $m_0$

1. $m_T \gets M_{coarse}$ // Khởi tạo với mặt nạ thô
2. $P_{fine}^{i,j} \gets 0$ cho mọi pixel $(i,j)$ // Khởi tạo xác suất mịn tích lũy
3. **for** $t = T, T-1, \dots, 1$ **do**
4. &nbsp;&nbsp;// **Bước 1: Dự đoán mặt nạ sạch và độ tin cậy**
5. &nbsp;&nbsp;$z_t \gets [I || m_t]$
6. &nbsp;&nbsp;$logits \gets f_\theta(z_t, t)$
7. &nbsp;&nbsp;$p_\theta^{i,j} \gets 2|\sigma(logits^{i,j}) - 0.5|$ // Điểm tin cậy (Confidence score)
8. &nbsp;&nbsp;// **Bước 2: Tính xác suất chuyển đổi ngược (reversed transition probability)**
9. &nbsp;&nbsp;$p_{c \to f}^{i,j} \gets \frac{p_\theta^{i,j} \cdot (\bar{\beta}_{t-1} - \bar{\beta}_t)}{1 - p_\theta^{i,j} \cdot \bar{\beta}_t}$ cho mọi $(i,j)$
10. &nbsp;&nbsp;$P_{fine}^{i,j} \gets P_{fine}^{i,j} + (1 - P_{fine}^{i,j}) \cdot p_{c \to f}^{i,j}$
11. &nbsp;&nbsp;// **Bước 3: Cập nhật mặt nạ**
12. &nbsp;&nbsp;**if** $t = 1$ **then**
13. &nbsp;&nbsp;&nbsp;&nbsp;$m_0^{i,j} \gets \sigma(logits^{i,j})$ // Bước cuối: deterministic
14. &nbsp;&nbsp;**else**
15. &nbsp;&nbsp;&nbsp;&nbsp;$\hat{m}_0^{i,j} \gets \mathbf{1}[logits^{i,j} \ge 0]$ // Dự đoán cứng (Hard prediction)
16. &nbsp;&nbsp;&nbsp;&nbsp;**for** mỗi pixel $(i,j)$ **do**
17. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$u^{i,j} \sim Uniform(0,1)$
18. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$m_{t-1}^{i,j} \gets \mathbf{1}[u < P_{fine}^{i,j}] \cdot \hat{m}_0^{i,j} + \mathbf{1}[u \ge P_{fine}^{i,j}] \cdot M_{coarse}^{i,j}$
19. &nbsp;&nbsp;&nbsp;&nbsp;**end for**
20. &nbsp;&nbsp;**end if**
21. **end for**
22. **return** $m_0$

---

## B. Tạo mặt nạ thô đa cấp (Multi-Level Coarse Mask Generation)

Không giống như SegRefiner gốc, phương pháp của chúng tôi xây dựng ba thành phần nhiễu bổ trợ từ ảnh RGB:

### 1. Cấp độ Vùng / Độ bất định ($M_{coarse}^{unc}$)
Sử dụng một bộ phân loại yếu (**Weak Classifier**) dựa trên mô hình GMM (Gaussian Mixture Model) được khớp trên các giá trị pixel RGB.
- Ước tính bản đồ độ bất định $M_{coarse}^{unc}$ bằng hàm Entropy của dự đoán GMM.
- Giúp xác định các vùng mà mô hình khó phân loại dựa trên đặc điểm hình ảnh.

### 2. Cấp độ Đối tượng ($M_{coarse}^{obj}$)
Trích xuất các thành phần liên thông (objects) $\{C_1, C_2, \dots, C_N\}$ từ $M_{fine}$.
- Tính độ bất định cấp đối tượng $U_k^{obj}$ bằng cách lấy trung bình độ bất định của các pixel trong đối tượng đó.
- Các đối tượng có độ bất định cao (thường là building nhỏ hoặc tương phản thấp) có khả năng bị loại bỏ hoàn toàn trong quá trình làm nhiễu, mô phỏng lỗi thiếu sót (missed objects) thực tế.

### 3. Cấp độ Đường biên ($M_{coarse}^{bnd}$)
Trích xuất thông tin đường biên từ ảnh vệ tinh bằng thuật toán **Canny**.
- Trong ảnh vệ tinh, ranh giới tòa nhà được xác định bởi sự đứt gãy màu sắc thay vì các đường cong hình học mịn. Do đó, cạnh từ ảnh RGB có giá trị hơn cạnh từ mặt nạ $M_{fine}$.

---

## C. Cải tiến Q-Sampling (Modified Q-Sampling)

Giai đoạn Q-sampling tạo ra mặt nạ nhiễu trung gian $m_t$ bằng cách kết hợp nhiễu cấp đối tượng và cấp pixel.

### Các thành phần tại bước thời gian $t$:
- **Uncertainty map:** $M_{coarse,t}^{unc} = \text{Erode}(U_{score}, n_{iter} = T - t)$
- **Object noise:** $M_{coarse,t}^{obj}(i,j) = 0$ nếu pixel thuộc đối tượng có độ bất định cao ($U_k^{obj} > 1 - \bar{\beta}_t$), ngược lại giữ $M_{fine}$.
- **Boundary noise:** $M_{coarse,t}^{bnd} = \text{Dilate}(M_{coarse}^{bnd}, n_{iter} = t)$

### Kết hợp nhiễu Super-pixel:
$M_{coarse,t}^{sp} = M_{coarse,t}^{bnd} \cup M_{coarse,t}^{unc}$

### Công thức trộn cuối cùng (PT 10):
$m_t^{i,j} = \tau^{i,j} \cdot M_{coarse,t}^{obj}(i,j) + (1 - \tau^{i,j}) \cdot M_{coarse,t}^{sp}(i,j)$

Với $\tau$ là biến ngẫu nhiên Bernoulli dựa trên lịch trình nhiễu $\bar{\beta}_t$. Việc thay thế $M_{fine}$ bằng $M_{coarse,t}^{obj}$ trong công thức này cho phép mô hình mô phỏng đồng thời việc thiếu tòa nhà, biên không chính xác và nhiễu do độ bất định gây ra.
