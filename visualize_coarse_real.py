"""
Visualization của Coarse Map KHỚP CHÍNH XÁC với q_sample hiện tại trong training.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy import ndimage as ndi
from skimage import measure
from sklearn.mixture import GaussianMixture

# ─── CẤU HÌNH (KHỚP VỚI segrefiner_semantic.py) ──────────────────────────────
DATA_ROOT = '/home/ubuntu/vy/Denoiser/OpenEarthMap_wo_xBD'
CITY = 'malopolskie'
NAME = 'malopolskie_10'
T_MAX = 6  # num_timesteps

# Thông số Canny Level 4 (khớp precompute_maps.py)
NEW_CANNY_LOW = 120
NEW_CANNY_HIGH = 250
MIN_EDGE_AREA = 500

# Thông số noise (khớp q_sample trong segrefiner_semantic.py)
TAU_UNC = 0.5           # dòng 419: tau_unc = 0.5
BND_KERNEL = lambda t: 2 * t + 3   # dòng 411: kernel = 3 * t_val + 5
# ──────────────────────────────────────────────────────────────────────────────

img_bgr = cv2.imread(f"{DATA_ROOT}/{CITY}/images/{NAME}.tif")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gt = (cv2.imread(f"{DATA_ROOT}/{CITY}/labels/{NAME}.tif", 0) == 1).astype(np.float32)

# Load/tính Uncertainty
unc_path = f"{DATA_ROOT}/{CITY}/uncertainty/{NAME}.npy"
if os.path.exists(unc_path):
    unc = np.load(unc_path)
else:
    pix = (img_rgb.astype(np.float32) / 255.0).reshape(-1, 3)
    idx = np.random.choice(len(pix), min(50_000, len(pix)), replace=False)
    gmm = GaussianMixture(2, covariance_type='full', random_state=42).fit(pix[idx])
    proba = gmm.predict_proba(pix)
    unc = (-np.sum(proba * np.log(proba + 1e-8), axis=1) / np.log(2)).reshape(img_rgb.shape[:2])

# Load/tính Edge (Level 4 cleanup)
edge_path = f"{DATA_ROOT}/{CITY}/edges/{NAME}.npy"
if os.path.exists(edge_path):
    edge = np.load(edge_path)
else:
    img_blur = cv2.GaussianBlur(img_rgb, (5, 5), 0)
    cR = cv2.Canny(img_blur[:,:,0], NEW_CANNY_LOW, NEW_CANNY_HIGH)
    cG = cv2.Canny(img_blur[:,:,1], NEW_CANNY_LOW, NEW_CANNY_HIGH)
    cB = cv2.Canny(img_blur[:,:,2], NEW_CANNY_LOW, NEW_CANNY_HIGH)
    edge_raw = cv2.bitwise_or(cv2.bitwise_or(cR, cG), cB)
    edge_conn = cv2.dilate(edge_raw, np.ones((3,3), np.uint8), iterations=1)
    nb, out, stats, _ = cv2.connectedComponentsWithStats(edge_conn, connectivity=8)
    res = np.zeros(edge_raw.shape, dtype=np.uint8)
    for j in range(1, nb):
        if stats[j, cv2.CC_STAT_AREA] >= MIN_EDGE_AREA:
            res[out == j] = 255
    edge = res.astype(np.float32) / 255.0

# Beta schedule — KHỚP CHÍNH XÁC với segrefiner_base.py:
# np.linspace(start=0.8, stop=0, num_timesteps=6)
betas_cumprod = np.linspace(0.8, 0.0, T_MAX)  # [0.8, 0.64, 0.48, 0.32, 0.16, 0.0]
# t=0 -> 0.8 (Sạch nhất), t=5 -> 0.0 (Bẩn nhất)

def q_sample_vis(t):
    """Mô phỏng chính xác q_sample trong segrefiner_semantic.py"""
    beta_b = betas_cumprod[t]

    # ── Object-level noise (Eq. 7: U_k^obj = mean unc của building) ────
    labeled = measure.label(gt > 0.5)
    instances = [(labeled == i) for i in range(1, labeled.max() + 1)]

    M_obj = np.zeros_like(gt)
    if len(instances) > 0:
        threshold = 1.0 / (1.0 + np.exp(-beta_b))  # sigmoid(beta_b)
        unc_scores = [unc[inst].mean() for inst in instances]
        kept_any = False
        for inst, u_k in zip(instances, unc_scores):
            if u_k <= threshold:   # U_k^obj <= sigmoid(beta_t) → giữ building
                M_obj[inst] = 1.0
                kept_any = True
        # Luôn giữ ít nhất 1 building chắc chắn nhất
        if not kept_any:
            best_idx = int(np.argmin(unc_scores))
            M_obj[instances[best_idx]] = 1.0

    # ── Boundary-level noise (Chỉ áp dụng cho nhà CÒN TỒN TẠI) ──────
    M_bnd = np.zeros_like(gt)
    kernel = BND_KERNEL(t)
    # Lấy vùng biên gốc
    M_bnd_raw = cv2.dilate(edge, np.ones((kernel, kernel), np.uint8))
    # Lấy vùng bao phủ của nhà hiện tại (M_obj)
    M_obj_area = cv2.dilate(M_obj, np.ones((kernel, kernel), np.uint8))
    # Chỉ giữ lại biên của nhà còn sống
    M_bnd = ((M_bnd_raw > 0.5) & (M_obj_area > 0.5)).astype(np.float32)

    dice = np.random.rand(*gt.shape)
    M_pixel_applied = M_obj.copy()
    M_pixel_applied[M_bnd > 0.5] = 1.0 - gt[M_bnd > 0.5]  # paper: 1 - M_fine

    # ── Uncertainty-level noise (UNC flip) ──────────────────────────
    M_unc_region = np.zeros_like(gt)
    tau_unc = 0.5
    unc_binary = (unc > tau_unc).astype(np.float32)
    
    n_erode = T_MAX - t
    if n_erode > 0 and unc_binary.any():
        # Erosion bằng max_pool2d ngược trong training:
        kernel_e = 2 * n_erode + 1
        # Dùng opencv erode
        M_unc_region = cv2.erode(unc_binary, np.ones((kernel_e, kernel_e), np.uint8))
    else:
        M_unc_region = unc_binary
    M_pixel_applied[M_unc_region > 0.5] = 1.0 - gt[M_unc_region > 0.5]

    # ── Final composite noise (Mixing Eq.11) ───────────────────────
    # tau ~ Bernoulli(beta)
    # tau=1 -> Chọn M_obj (Sạch hơn), tau=0 -> Chọn M_pixel (Bẩn hơn)
    tau = (np.random.rand(*gt.shape) < beta_b).astype(np.float32)
    m_t = tau * M_obj + (1 - tau) * M_pixel_applied
    
    return (m_t >= 0.5).astype(np.float32)


# Visualize: chỉ hiện t=0 đến t=5 (khớp với num_timesteps=6 trong Training)
STEPS = list(range(T_MAX))  # [0, 1, 2, 3, 4, 5]
fig, axes = plt.subplots(len(STEPS), 4, figsize=(20, 3.5 * len(STEPS)))
fig.patch.set_facecolor('#0d1117')

for i, t in enumerate(STEPS):
    coarse = q_sample_vis(t)
    overlay = img_rgb.copy()
    overlay[coarse == 1] = overlay[coarse == 1] * 0.6 + np.array([255, 0, 0]) * 0.4
    diff = np.zeros((*gt.shape, 3))
    diff[np.logical_and(coarse == 1, gt == 1)] = [1, 1, 1]   # Đúng (trắng)
    diff[np.logical_and(coarse == 1, gt == 0)] = [1, 0, 0]   # FP (đỏ)
    diff[np.logical_and(coarse == 0, gt == 1)] = [0, 0.5, 1] # FN (xanh)

    axes[i, 0].imshow(coarse, cmap='gray')
    axes[i, 0].set_title(f"Coarse t={t}  (β={betas_cumprod[t]:.2f})", color='white')
    axes[i, 1].imshow(overlay)
    axes[i, 1].set_title(f"Overlay t={t}", color='white')
    axes[i, 2].imshow(diff)
    axes[i, 2].set_title("Red=FP | Blue=FN", color='white')
    axes[i, 3].imshow(gt, cmap='gray')
    axes[i, 3].set_title("Ground Truth", color='white')
    for ax in axes[i]: ax.axis('off')

plt.tight_layout()
out = f"/home/ubuntu/vy/Denoiser/SegRefiner/train_coarse_vis_{NAME}.png"
plt.savefig(out, facecolor='#0d1117', dpi=100)
print(f"✅ Saved: {out}")
