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
CITY = 'melbourne'
NAME = 'melbourne_10'
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

    # ── Equation 7: Object-level noise (Xóa vật thể) ──────────────────
    labeled = measure.label(gt > 0.5)
    instances = [(labeled == i) for i in range(1, labeled.max() + 1)]
    M_obj = np.zeros_like(gt)
    if len(instances) > 0:
        threshold = beta_b  # Bài báo dùng trực tiếp beta_t
        for inst_mask in instances:
            u_k = unc[inst_mask].mean()
            if u_k <= threshold:   # Giữ lại nếu độ tin cậy cao hơn ngưỡng
                M_obj[inst_mask] = 1.0
        # Đảm bảo không bị trống hoàn toàn
        if M_obj.sum() == 0:
            best_idx = np.argmin([unc[inst].mean() for inst in instances])
            M_obj[instances[best_idx]] = 1.0

    # ── Equation 8: Boundary-level noise (ÉP NHIỄU Ở T=0) ─────────────
    # Thay vì n = t, dùng n = t + 1 để t=0 vẫn bị dãn biên 1 vòng
    kernel_b = np.ones((3, 3), np.uint8)
    M_bnd = cv2.dilate(edge, kernel_b, iterations=t + 1)

    # ── Equation 6: Uncertainty-level noise (Y chang bài báo) ─────────
    tau_unc = 0.5
    unc_binary = (unc > tau_unc).astype(np.uint8)
    n_iter_unc = T_MAX - t
    if n_iter_unc > 0:
        kernel_u = np.ones((3, 3), np.uint8)
        M_unc = cv2.erode(unc_binary, kernel_u, iterations=n_iter_unc)
    else:
        M_unc = unc_binary
    
    # ── Equation 9: Combined Super-pixel noise (Boundary U Uncertainty)
    M_sp = ((M_bnd > 0.5) | (M_unc > 0.5)).astype(np.float32)

    # Logic đảo ngược pixel (Hole punching 1-gt)
    M_pixel_applied = M_obj.copy()
    # Chỉ áp dụng nhiễu tại vùng M_sp
    M_pixel_applied[M_sp > 0.5] = 1.0 - gt[M_sp > 0.5]

    # ── Final Mixing (Bernoulli with beta_t) ─────────────────────────
    # t=0 -> beta=0.8 (Sạch), t=5 -> beta=0.0 (Bẩn)
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
