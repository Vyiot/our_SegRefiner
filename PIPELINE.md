# SegRefiner – Phân Tích Full Pipeline (Train & Val)

## 1. Cấu Hình Hiện Tại

```python
# segrefiner_oem_base.py
num_timesteps  T = 6
betas_cumprod    = linspace(0.8, 0, T=6)
               = [0.8, 0.64, 0.48, 0.32, 0.16, 0.0]
#   Index:        0     1     2     3     4     5
#   Ý nghĩa:      ← Sạch nhất              Bẩn nhất →

# exp8_all.py
noise_components: use_obj=True, use_bnd=True, use_unc=True
```

**Quy ước:**  
- `t_val` = index (0 → 5)  
- `t_val = 0` → β̄ = 0.8 → **Sạch nhất** (80% lấy từ GT)  
- `t_val = 5` → β̄ = 0.0 → **Bẩn nhất** (100% coarse, không có GT)

---

## 2. Ba Loại Nhiễu (Multi-Level Coarse Masks)

Paper sử dụng 3 loại bản đồ nhiễu, mỗi cái mô phỏng một loại lỗi khác nhau của pseudo label:

| Tên | Công thức | Mô phỏng lỗi gì |
|-----|-----------|-----------------|
| **M_unc_coarse** | Eq.1: `H[W(x_rgb)] / σ` | Vùng pixel mờ hồ, khó phân loại (bóng đổ, mái tối...) |
| **M_obj_coarse** | Eq.4: `U_k^obj` per pixel | Pseudo label bị mất hẳn cả tòa nhà (false negative) |
| **M_bnd_coarse** | Eq.5: `Canny(x_rgb)` | Pseudo label bị lem nhòe đường biên tòa nhà |

> **Lưu ý:** `M_unc_coarse` và `M_bnd_coarse` được **tính sẵn** bằng `precompute_maps.py`  
> và lưu vào `{city}/uncertainty/*.npy` và `{city}/edges/*.npy`.

---

## 3. Luồng TRAINING

### 3.1. DataLoader (`LoadOEMCoarseMasks` + `LoadObjectData`)

```
RGB ảnh vệ tinh (x_rgb)  +  GT mask (M_fine)
         │
         ├─► Load M_unc_coarse  từ uncertainty/{img}.npy   (Eq.1)
         ├─► Load M_bnd_coarse  từ edges/{img}.npy         (Eq.5)
         └─► Crop 256×256 patch (LoadObjectData)
```

**Output truyền vào model:**
- `object_img`          → RGB ảnh (B, 3, H, W)
- `object_gt_masks`     → **M_fine = GT** (B, 1, H, W)  ← dùng trong q_sample
- `object_coarse_masks` → Coarse mask (không dùng trong q_sample mới)
- `object_unc_map`      → M_unc_coarse (B, H, W)
- `object_edge_map`     → M_bnd_coarse (B, H, W)

> **Quan trọng:** Trong Training, **M_fine = GT mask** (không phải pseudo label).  
> Mô hình học từ GT sạch, `q_sample` sẽ corrupt GT để tạo `m_t`.

---

### 3.2. `forward_train` (`segrefiner_base.py`)

```python
target  = GT  = M_fine          # x_start
t       = Uniform({0,1,2,3,4,5})  # timestep ngẫu nhiên cho mỗi sample

x_t = q_sample(target, t)       # Tạo noisy mask m_t từ GT

z_t = concat(RGB, x_t)          # (B, 4, H, W)
pred = denoise_model(z_t, t)    # Predict lại GT sạch

loss = loss_mask(pred, GT) + loss_texture(pred, GT)
```

---

### 3.3. `q_sample` – Modified Q-Sampling (Eq.2–11)

> **File:** `segrefiner_semantic.py`

Với mỗi sample có timestep `t_val` (index 0–5):

---

#### Eq.2 – FindObject(M_fine)
```
{C₁, C₂, ..., Cₙ} = connected_components(GT > 0.5)
→ ndi.label(gt_np)
```
Tìm từng tòa nhà riêng lẻ trong GT mask.

---

#### Eq.3 – Tính Uncertainty Từng Tòa Nhà
```
U_k^obj = mean( M_unc_coarse[pixels ∈ C_k] )
→ unc_np[mask_k].mean()
```

---

#### Eq.7 – Object-Level Noise: M_obj_coarse_t
```
M_obj_t(i,j) = { 0          nếu (i,j)∈C_k  VÀ  U_k^obj > β̄_t
               { GT(i,j)    ngược lại
```
**Xóa cả tòa nhà** nếu tòa nhà đó có uncertainty cao hơn β̄_t.

| t_val | β̄_t | Tòa nhà bị xóa khi |
|-------|------|--------------------|
| 5 (bẩn nhất) | 0.0 | U_k > 0.0 → Xóa MỌI tòa nhà |
| 3 | 0.32 | U_k > 0.32 |
| 0 (sạch nhất) | 0.8 | U_k > 0.8 → Chỉ xóa tòa nhà RẤT không chắc |

---

#### Eq.8 – Boundary-Level Noise: M_bnd_coarse_t
```
M_bnd_t = Dilate(M_bnd_coarse, n_iter = t_val)
→ max_pool2d(edge_map, kernel = 2*t_val + 1)
```
**Làm dày đường biên** theo mức độ t.

| t_val | n_iter | Độ dày biên |
|-------|--------|-------------|
| 5 (bẩn) | 5 | Rất dày (±5 pixel) |
| 0 (sạch) | 0 | Không có biên |

---

#### Eq.6 – Uncertainty-Level Noise: M_unc_coarse_t
```
unc_binary   = 1[M_unc_coarse > τ_unc]   (τ_unc = 0.5)
M_unc_t      = Erode(unc_binary, n_iter = T - t_val)
→ ~max_pool2d(~unc_binary, kernel = 2*(T-t_val) + 1)
```
**Erode vùng uncertain** – t càng nhỏ (sạch hơn), erosion càng mạnh.

| t_val | n_erode = T-t | Vùng uncertain còn lại |
|-------|--------------|------------------------|
| 5 (bẩn) | 1 | Erosion nhẹ → Giữ nhiều vùng uncertain |
| 0 (sạch) | 6 | Erosion mạnh → Chỉ còn lõi uncertain |

---

#### Eq.9 – Super-Pixel Noise Region
```
M_Sp_t = M_bnd_t ∪ M_unc_t
→ (M_bnd_t + M_unc_t) > 0
```

---

#### Eq.10 – Pixel-Applied Coarse Mask
```
M_pixel_applied_t(i,j) = { 1 - GT(i,j)   nếu M_Sp_t(i,j) = 1   ← Lật pixel
                          { GT(i,j)        nếu M_Sp_t(i,j) = 0   ← Giữ GT
```

---

#### Eq.11 – Final Q-Sampling → Tạo m_t
```
τ(i,j)  = 1[Uniform(0,1) < β̄_t]    ← Bernoulli per pixel

m_t(i,j) = τ · M_obj_t(i,j)  +  (1-τ) · M_pixel_applied_t(i,j)
```
Mỗi pixel chọn ngẫu nhiên từ 1 trong 2 nguồn nhiễu:
- `τ=1` (xác suất β̄_t): Lấy từ M_obj (có thể thiếu tòa nhà)
- `τ=0` (xác suất 1−β̄_t): Lấy từ M_pixel_applied (pixel biên/uncertain bị lật)

**Kết quả m_t theo t_val:**

| t_val | β̄_t | m_t trông như thế nào |
|-------|------|----------------------|
| 5 (bẩn) | 0.0 | τ=0 mọi pixel → 100% M_pixel_applied (biên + uncertain bị lật) |
| 3 | 0.32 | Trộn 32% M_obj + 68% M_pixel_applied |
| 0 (sạch) | 0.8 | 80% M_obj (gần GT), 20% M_pixel_applied |

---

### 3.4. Model Học Gì?

```
Input:  concat(RGB, m_t)  →  DenoiseUNet(t)  →  Predict GT sạch
```

Model học cách **đọc thông tin từ RGB** để phân biệt lỗi do nhiễu ra khỏi cấu trúc thật của tòa nhà. Sau khi train xong, không cần `unc_map` hay `edge_map` nữa.

---

## 4. Luồng VALIDATION (Inference – Algorithm 2)

> **File:** `segrefiner_semantic.py` → `simple_test_semantic`

### Không có GT, không có q_sample. Input chỉ có:
- `img`: RGB ảnh vệ tinh
- `coarse_masks`: **Pseudo label** (nhãn giả từ upstream model)

---

### Giai Đoạn 1 – Global Refinement (256×256)

```
Pseudo Label
     │  resize xuống 256×256
     ▼
indices = [5, 4, 3, 2, 1]   ← Chạy từ bẩn (t=5) về sạch (t=1)
     │
  p_sample(t=5): model nhìn (RGB_256, pseudo_256) → trả về x₄
  p_sample(t=4): model nhìn (RGB_256, x₄)         → trả về x₃
  p_sample(t=3): ...                               → x₂
  p_sample(t=2): ...                               → x₁
  p_sample(t=1): ...                               → x₀ (global refined)
     │
  interpolate lên 1024×1024
     ↓
  ori_size_mask (1024×1024)
```

---

### Giai Đoạn 2 – Local Refinement (1024×1024)

```
ori_size_mask + fine_probs (độ tự tin của từng pixel)
     │
  Tìm vùng low-confidence: fine_prob < fine_prob_thr × max(fine_probs)
  Cắt patches 256×256 tại các vùng đó
  NMS để loại trùng
     │
  p_sample(t=0): model tinh chỉnh từng patch ở độ phân giải gốc
     │
  Dán patches lại vào ori_size_mask
     ↓
  Final Mask (1024×1024)  ← So sánh với GT → tính mIoU
```

---

## 5. So Sánh Train vs Val

| Khía cạnh | Training | Validation |
|-----------|----------|------------|
| **M_fine** | GT mask | Pseudo label |
| **Noise source** | `q_sample` (Eq.6–11) | Không có noise |
| **t_val** | Random ∈ {0,1,2,3,4,5} | Tuần tự: 5→4→3→2→1→0 |
| **Mục tiêu model** | Predict GT từ `(RGB, m_t)` | Predict refined mask từ `(RGB, pseudo)` |
| **Loss** | loss_mask + loss_texture | Không có loss → tính IoU với GT |
| **unc_map / edge_map** | ✅ Dùng trong q_sample | ❌ Không dùng |
| **Kích thước** | 256×256 patches | 256×256 global → 1024×1024 local |

---

## 6. Tóm Tắt Tại Sao Val Dùng t=5→0?

Pseudo label chứa các lỗi tương tự "trạng thái t=5" (bẩn nhất):
- Một số tòa nhà bị mất
- Đường biên bị lem nhòe
- Vùng uncertain bị phân loại sai

Model dùng 6 bước (t=5→4→3→2→1→0) để **dần dần "rửa sạch"** pseudo label dựa trên thông tin RGB, bước cuối (t=0) cho ra kết quả tinh chỉnh nhất.
