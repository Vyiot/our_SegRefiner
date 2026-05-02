# Changelog: Multi-Level Mask Denoiser (OEM Building)

Ghi lại tất cả các file đã **thêm mới** và **chỉnh sửa** trong quá trình implement ý tưởng vào SegRefiner.

---

## Tổng quan thay đổi

```
mmdet/
├── models/utils/
│   └── noise_utils.py           [NEW]
├── datasets/
│   ├── oem_building.py          [NEW]
│   ├── __init__.py              [MODIFIED]
│   └── pipelines/
│       ├── loading.py           [MODIFIED - append]
│       └── __init__.py          [MODIFIED]
├── models/detectors/
│   └── segrefiner_base.py       [MODIFIED]
├── core/evaluation/
│   ├── oem_eval_hook.py         [NEW]
│   └── __init__.py              [MODIFIED]
apis/
│   └── train.py                 [MODIFIED - wire OEMBuildingEvalHook]
configs/segrefiner/
├── segrefiner_oem_base.py       [NEW]
├── exp1_base.py                 [NEW]
├── exp2_obj.py                  [NEW]
├── exp3_bnd.py                  [NEW]
├── exp4_unc.py                  [NEW]
├── exp5_obj_bnd.py              [NEW]
├── exp6_obj_unc.py              [NEW]
├── exp7_bnd_unc.py              [NEW]
└── exp8_all.py                  [NEW]
```

---

## Chi tiết từng file

---

### [NEW] `mmdet/models/utils/noise_utils.py`

**Mục đích:** Chứa 3 hàm tiện ích tạo nhiễu đa cấp.

#### Hàm A: `compute_gmm_uncertainty(img_rgb, n_components=3)`
```python
# Input : img_rgb (H,W,3) uint8
# Output: uncertainty_map (H,W) float32 [0,1]
# Logic : GMM n_components=3, tính Entropy, chuẩn hóa về [0,1]
# Lý do n=3: phân biệt được 3 nhóm màu phổ biến trong ảnh vệ tinh
#   - Cụm 1: Nhà sáng (mái ngói đỏ, bê tông sáng)
#   - Cụm 2: Nhà tối (bóng râm, mái tối)
#   - Cụm 3: Nền (cỏ, đất, đường)
# => Entropy cao tập trung hơn ở biên tòa nhà thay vì lan khắp ảnh (như n=2)
gmm = GaussianMixture(n_components=3, covariance_type='full', max_iter=50)
log_proba = gmm.predict_proba(pixels)
entropy = -np.sum(log_proba * np.log(log_proba + eps), axis=1)
uncertainty = entropy / np.log(n_components)   # chuẩn hóa bằng log(3)
```

#### Hàm B: `extract_building_instances(gt_mask)`
```python
# Input : gt_mask (H,W) uint8 0/1  <- đây là GROUND TRUTH
# Output: list[dict]  mỗi dict gồm mask, area, bbox
# Logic : scipy.ndimage.label để tìm vùng liên thông
labeled, n = ndimage.label(gt_mask)
for id in range(1, n+1):
    single_mask = (labeled == id)
    instances.append({'mask': single_mask, 'area': ..., 'bbox': ...})
```

#### Hàm C: `get_rgb_edges_gpu(img_rgb_tensor, low=0.1, high=0.2)`
```python
# Input : Tensor (B,3,H,W) hoặc (3,H,W), float [0,1], GPU
# Output: Tensor (B,1,H,W) hoặc (1,H,W), 0.0/1.0
# Logic : kornia.color.rgb_to_grayscale -> kornia.filters.canny
grayscale = kornia.color.rgb_to_grayscale(img_rgb_tensor)
_, edge_map = kornia.filters.canny(grayscale, low_threshold=0.1, high_threshold=0.2)
```

---

### [NEW] `mmdet/datasets/oem_building.py`

**Mục đích:** Dataset class cho OpenEarthMap Building.

**Class:** `OEMBuildingDataset(CustomDataset)`

**Key logic:**
- Đọc danh sách ảnh từ `train.txt` / `val.txt`
- In ra log: `INFO - Train set: 3000 images` / `INFO - Val set: 500 images`
- Label của OpenEarthMap: class 1 = Building → chuyển về binary mask `(label == 1)`
- Tự động tìm thư mục `city` bằng cách khớp tiền tố tên file với thư mục trong `data_root`

---

### [MODIFIED] `mmdet/datasets/pipelines/loading.py`

**Thêm vào cuối file:** Class `LoadOEMCoarseMasks`

**Constructor:**
```python
LoadOEMCoarseMasks(
    use_obj=True,            # Bật Object-level noise
    use_bnd=True,            # Bật Boundary-level noise
    use_unc=True,            # Bật Uncertainty-level noise
    obj_unc_threshold=0.5,   # Ngưỡng xóa tòa nhà
    canny_low=0.1,           # Ngưỡng thấp Canny (cho ảnh RGB [0,1])
    canny_high=0.2,          # Ngưỡng cao Canny
    bnd_dilate_ksize=5,      # Kernel giãn biên
    test_mode=False          # True = không tạo noise
)
```

**Luồng xử lý `_generate_multilevel_noise(img, gt_mask)`:**
1. `use_unc or use_obj` → `compute_gmm_uncertainty(img)` → `uncertainty_map`
2. `use_obj` → `extract_building_instances(gt_mask)` → xóa tòa nhà có `mean_unc > 0.5`
3. `use_unc` → flip ngẫu nhiên pixel với xác suất `= uncertainty * 0.5`
4. `use_bnd` → `get_rgb_edges_gpu(img/255)` → dilate → flip biên xác suất 0.4
5. Nếu tất cả False → fallback `modify_boundary(gt_255)` (gốc SegRefiner)

**Output vào `results`:**
```python
results['gt_masks']     = BitmapMasks([gt_mask], H, W)
results['coarse_masks'] = BitmapMasks([coarse], H, W)
results['mask_fields']  = ['gt_masks', 'coarse_masks']
```

---

### [MODIFIED] `mmdet/datasets/__init__.py`

Thêm 2 dòng:
```python
from .oem_building import OEMBuildingDataset   # import
'OEMBuildingDataset'                           # vào __all__
```

---

### [MODIFIED] `mmdet/datasets/pipelines/__init__.py`

Thêm vào import và `__all__`:
```python
LoadOEMCoarseMasks    # từ loading.py
```

---

### [MODIFIED] `mmdet/models/detectors/segrefiner_base.py`

**Thay đổi trong `_diffusion_init`:** Thêm đọc `noise_components` từ `diffusion_cfg`.

```python
# Trước (gốc)
self.num_timesteps = self.betas_cumprod.shape[0]

# Sau (mới) - thêm vào cuối _diffusion_init
noise_cfg = diffusion_cfg.get('noise_components', {})
self.use_obj = noise_cfg.get('use_obj', False)
self.use_bnd = noise_cfg.get('use_bnd', False)
self.use_unc = noise_cfg.get('use_unc', False)
```

> **Lưu ý:** `q_sample` không thay đổi vì logic tạo nhiễu đã được đưa vào `LoadOEMCoarseMasks` ở tầng Data Pipeline. `self.use_*` được lưu lại để có thể dùng cho logging hoặc mở rộng sau này.

---

### [NEW] Config Files

| File | Obj | Bnd | Unc | Mô tả |
|:---|:---:|:---:|:---:|:---|
| `segrefiner_oem_base.py` | - | - | - | Base config, kế thừa bởi tất cả exp |
| `exp1_base.py` | ❌ | ❌ | ❌ | SegRefiner gốc (modify_boundary) |
| `exp2_obj.py` | ✅ | ❌ | ❌ | Chỉ Object noise |
| `exp3_bnd.py` | ❌ | ✅ | ❌ | Chỉ Boundary noise |
| `exp4_unc.py` | ❌ | ❌ | ✅ | Chỉ Uncertainty noise |
| `exp5_obj_bnd.py` | ✅ | ✅ | ❌ | Object + Boundary |
| `exp6_obj_unc.py` | ✅ | ❌ | ✅ | Object + Uncertainty |
| `exp7_bnd_unc.py` | ❌ | ✅ | ✅ | Boundary + Uncertainty |
| `exp8_all.py` | ✅ | ✅ | ✅ | **Phương pháp đầy đủ** |

---

## Lệnh chạy

```bash
# Chạy thí nghiệm Exp 1 (baseline)
python tools/train.py configs/segrefiner/exp1_base.py \
    --work-dir work_dirs/exp1_base

# Chạy thí nghiệm Exp 8 (đầy đủ)
python tools/train.py configs/segrefiner/exp8_all.py \
    --work-dir work_dirs/exp8_all
```

---

## Phụ thuộc cần cài thêm

```bash
pip install scikit-learn   # cho GaussianMixture
pip install scipy          # cho ndimage.label
pip install kornia         # cho Canny GPU
```
