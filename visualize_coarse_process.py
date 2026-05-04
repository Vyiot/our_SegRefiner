"""
visualize_coarse_process.py
Mô phỏng quá trình tạo Coarse Mask (nhiễu) trong Training của SegRefiner.
"""
import cv2, numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

# ── Cấu hình ──
CITY = 'melbourne'
NAME = 'melbourne_48'
DATA_ROOT = '/home/ubuntu/vy/Denoiser/OpenEarthMap_wo_xBD'

# Hyperparams từ paper/code train
BETA_VAL = 0.5   # Ngưỡng xóa object (Eq. 7)
T_MAX    = 12    # Timestep tối đa (theo config train)
T_VAL    = 12    # Visualize tại bước nhiễu nặng nhất

# ── Load data ──
img_path  = f"{DATA_ROOT}/{CITY}/images/{NAME}.tif"
gt_path   = f"{DATA_ROOT}/{CITY}/labels/{NAME}.tif"
unc_path  = f"{DATA_ROOT}/{CITY}/uncertainty/{NAME}.npy"
edge_path = f"{DATA_ROOT}/{CITY}/edges/{NAME}.npy"

img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
gt  = (cv2.imread(gt_path, 0) == 1).astype(np.float32)

# Fallback if precomputed maps not found
if not os.path.exists(unc_path):
    print("Computing GMM on-the-fly for visualization...")
    from sklearn.mixture import GaussianMixture
    K = 2
    pix = (img.astype(np.float32) / 255.0).reshape(-1, 3)
    idx = np.random.choice(len(pix), min(50_000, len(pix)), replace=False)
    gmm = GaussianMixture(K, covariance_type='full', random_state=42).fit(pix[idx])
    proba = gmm.predict_proba(pix)
    unc = (-np.sum(proba * np.log(proba + 1e-8), axis=1) / np.log(K)).reshape(img.shape[:2])
else:
    unc = np.load(unc_path)

if not os.path.exists(edge_path):
    print("Computing Color Canny on-the-fly for visualization...")
    cR = cv2.Canny(img[:,:,0], 50, 150)
    cG = cv2.Canny(img[:,:,1], 50, 150)
    cB = cv2.Canny(img[:,:,2], 50, 150)
    edge = cv2.bitwise_or(cv2.bitwise_or(cR, cG), cB).astype(np.float32) / 255.0
else:
    edge = np.load(edge_path)

# ══════════════════════════════════════════════════════════════════════════════
# MÔ PHỎNG LOGIC TRONG segrefiner_semantic.py
# ══════════════════════════════════════════════════════════════════════════════

# 1. Object-level noise (Xóa cả tòa nhà)
labeled, num_objs = ndi.label(gt)
m_obj = gt.copy()
if num_objs > 0:
    for i in range(1, num_objs + 1):
        mask_i = (labeled == i)
        u_i = unc[mask_i].mean()
        if u_i > BETA_VAL:
            m_obj[mask_i] = 0.0

# 2. Boundary-level noise (Dilation của Canny)
# t=3 -> dilate kernel size = 2*t + 1 = 7
k_size = 2 * T_VAL + 1
m_bnd = cv2.dilate(edge, np.ones((k_size, k_size), np.uint8))

# 3. Uncertainty-level noise (Erosion của Entropy map)
# t=3 -> T-t = 6-3 = 3 iterations
thr_unc = 0.3
bin_unc = (unc > thr_unc).astype(np.float32)
m_unc = cv2.erode(bin_unc, np.ones((3, 3), np.uint8), iterations=(T_MAX - T_VAL))

# 4. Final Flip (Eq. 9)
# Pixel bị đảo ngược nếu thuộc m_obj, m_bnd hoặc m_unc
m_noise = np.logical_or(np.logical_or(m_bnd > 0.5, m_unc > 0.5), (gt != m_obj)).astype(np.float32)
coarse_mask = gt.copy()
coarse_mask[m_noise == 1] = 1 - coarse_mask[m_noise == 1]

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
fig.patch.set_facecolor('#0d1117')

def show(ax, data, title, cmap=None):
    ax.imshow(data, cmap=cmap)
    ax.set_title(title, color='white', fontsize=15, fontweight='bold')
    ax.axis('off')

# Hàng 1: Các thành phần tạo noise
show(axes[0, 0], img, "Original Image")
show(axes[0, 1], gt,  "GT Building Mask", "gray")
show(axes[0, 2], m_bnd, f"Boundary Noise (t={T_VAL})\nDilated Canny", "magma")
show(axes[0, 3], m_unc, f"Uncertainty Noise (t={T_VAL})\nEroded GMM Map", "magma")

# Hàng 2: Kết hợp
show(axes[1, 0], m_obj, "Object Noise (Eq. 7)\nBuildings with high U removed", "gray")
show(axes[1, 1], m_noise, "Total Noise Region\n(Pixels to be flipped)", "inferno")
show(axes[1, 2], coarse_mask, "FINAL COARSE MASK\n(Input for Model training)", "gray")

# So sánh Coarse vs GT (Red = lỗi, Blue = đúng)
diff = np.zeros((*gt.shape, 3))
diff[np.logical_and(coarse_mask == 1, gt == 1)] = [1, 1, 1] # Correct FG
diff[np.logical_and(coarse_mask == 1, gt == 0)] = [1, 0, 0] # False Positive (Noise)
diff[np.logical_and(coarse_mask == 0, gt == 1)] = [0, 0.5, 1] # False Negative (Noise)
show(axes[1, 3], diff, "Comparison: Coarse vs GT\nRed=Added, Blue=Removed", None)

plt.tight_layout()
out_file = "/home/ubuntu/vy/Denoiser/SegRefiner/coarse_visualized.png"
plt.savefig(out_file, facecolor='#0d1117')
print(f"Đã lưu visualization: {out_file}")
