"""
Visualization của Coarse Map ÁP DỤNG ĐÚNG CÁC CÔNG THỨC BÀI BÁO (Eq. 1–11).

Eq. 1 : M_unc_coarse  = H(W(x_rgb)) / σ  ∈ [0,1]   (GMM entropy, normalized)
Eq. 5 : M_bnd_coarse  = Canny(x_rgb)                (raw Canny, no dilation)
Eq. 6 : M_unc_t       = Erode( 1[M_unc > τ], n=T−t )
Eq. 7 : M_obj_t       = giữ C_k nếu U_k^obj <= β_t, xóa nếu > β_t
Eq. 8 : M_bnd_t       = Dilate( M_bnd, n=t )        (n=0 khi t=0 → KHÔNG dilate)
Eq. 9 : M_sp_t        = M_bnd_t ∪ M_unc_t
Eq.10 : M_pixel_t(i,j)= 1−M_fine nếu (i,j)∈M_sp, else M_fine
Eq.11 : m_t           = τ·M_obj_t + (1−τ)·M_pixel_t,  τ ~ Bernoulli(β_t)
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage import measure
from sklearn.mixture import GaussianMixture

# ─── CẤU HÌNH ────────────────────────────────────────────────────────────────
DATA_ROOT = '/home/ubuntu/vy/Denoiser/OpenEarthMap_wo_xBD'
CITY      = 'houston'
NAME      = 'houston_9'
T_MAX     = 6          # num_timesteps — betas_cumprod có 6 phần tử

# Beta schedule chuẩn bài báo: t=0 → 0.8 (sạch nhất), t=5 → 0.0 (bẩn nhất)
# β̄_t = linspace(0.8, 0.0, T)
betas_cumprod = np.linspace(0.8, 0.0, T_MAX)

# Ngưỡng tau_unc (Eq.6)
TAU_UNC = 0.5

# ─── ABLATION FLAGS (Bật/Tắt nhiễu) ──────────────────────────────────────────
USE_OBJ = True   # Eq.7: Object-level deletion
USE_BND = True   # Eq.8: Boundary dilation
USE_UNC = True   # Eq.6: Uncertainty erosion
# ──────────────────────────────────────────────────────────────────────────────

# ─── ĐỌC DỮ LIỆU ─────────────────────────────────────────────────────────────
img_bgr = cv2.imread(f"{DATA_ROOT}/{CITY}/images/{NAME}.tif")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gt      = (cv2.imread(f"{DATA_ROOT}/{CITY}/labels/{NAME}.tif", 0) == 1).astype(np.float32)

# ── Eq. 1: M_unc_coarse = H(W(x_rgb)) / σ ───────────────────────────────────
# Load precomputed nếu có, tính lại nếu chưa có
unc_path = f"{DATA_ROOT}/{CITY}/uncertainty/{NAME}.npy"
if os.path.exists(unc_path):
    unc = np.load(unc_path)   # đã là entropy [0,1]
else:
    pix = (img_rgb.astype(np.float32) / 255.0).reshape(-1, 3)
    idx = np.random.choice(len(pix), min(50_000, len(pix)), replace=False)
    gmm = GaussianMixture(2, covariance_type='full', random_state=42).fit(pix[idx])
    proba   = gmm.predict_proba(pix)
    entropy = (-np.sum(proba * np.log(proba + 1e-8), axis=1) / np.log(2))
    # Chuẩn hóa về [0,1] bằng σ (max entropy của 2-component GMM = 1 bit)
    unc = entropy.reshape(img_rgb.shape[:2]).astype(np.float32)
    unc = np.clip(unc, 0.0, 1.0)   # đã ∈ [0,1] vì chia log2

# ── Eq. 5: M_bnd_coarse = Canny(x_rgb) — RAW, không dilate ──────────────────
edge_path = f"{DATA_ROOT}/{CITY}/edges/{NAME}.npy"
if os.path.exists(edge_path):
    edge = np.load(edge_path)   # float32 ∈ {0, 1}
else:
    img_blur = cv2.GaussianBlur(img_rgb, (5, 5), 0)
    gray     = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    edge_raw = cv2.Canny(gray, 50, 150)
    edge     = (edge_raw.astype(np.float32) / 255.0)

# ─── HÀM Q_SAMPLE ────────────────────────────────────────────────────────────
def q_sample_vis(t):
    """
    Áp dụng ĐÚNG Eq. 6-11 của bài báo.

    Lịch betas: betas_cumprod = linspace(0.8→0.0, T=6)
        t=0 → β̄=0.8  (sạch nhất — ít nhiễu)
        t=5 → β̄=0.0  (bẩn nhất — nhiều nhiễu)
    """
    beta_b = betas_cumprod[t]   # β̄_t

    # ── Eq. 7: M_obj_t — Xóa/giữ object theo U_k^obj vs β_t ────────────────
    # M_fine = gt (ground truth)
    labeled   = measure.label(gt > 0.5)
    M_obj = gt.copy()
    if USE_OBJ:
        labeled   = measure.label(gt > 0.5)
        instances = [(labeled == i) for i in range(1, labeled.max() + 1)]
        M_obj_new = np.zeros_like(gt)
        if len(instances) > 0:
            unc_scores = [unc[inst].mean() for inst in instances]
            kept_any = False
            for inst_mask, u_k in zip(instances, unc_scores):
                if u_k <= beta_b:       # U_k^obj ≤ β_t → giữ lại
                    M_obj_new[inst_mask] = 1.0
                    kept_any = True
            
            # Fallback: Nếu xóa sạch thì giữ lại 1 cái tốt nhất để vis cho đẹp
            if not kept_any:
                best_idx = int(np.argmin(unc_scores))
                M_obj_new[instances[best_idx]] = 1.0
            M_obj = M_obj_new

    # ── Eq. 8: M_bnd_t (Dilate Canny Edge chuẩn bài báo: n_iter = t) ──────
    M_bnd = np.zeros_like(gt)
    if USE_BND:
        n_bnd = t   # t=0 -> 0 (sạch), t=5 -> 5 (bẩn)
        if n_bnd > 0:
            kernel_bnd = np.ones((3, 3), np.uint8)
            M_bnd = cv2.dilate((edge > 0.5).astype(np.uint8), kernel_bnd, iterations=n_bnd).astype(np.float32)
        else:
            M_bnd = (edge > 0.5).astype(np.float32)

    # ── Eq. 6: M_unc_t (Erode vùng Uncertain theo T-t) ─────────────────────
    M_unc = np.zeros_like(gt)
    if USE_UNC:
        unc_binary = (unc > TAU_UNC).astype(np.uint8)
        n_erode = T_MAX - t
        if n_erode > 0:
            kernel_unc = np.ones((3, 3), np.uint8)
            M_unc = cv2.erode(unc_binary, kernel_unc, iterations=n_erode).astype(np.float32)
        else:
            M_unc = unc_binary.astype(np.float32)

    # ── Eq. 9: M_sp_t = M_bnd_t ∪ M_unc_t ───────────────────────────────────
    M_sp = ((M_bnd > 0.5) | (M_unc > 0.5)).astype(np.float32)

    # ── Eq. 10: M_pixel-applied_t — base = M_fine (gt), flip vùng M_sp ──────
    # Paper: M_pixel_applied(i,j) = 1-M_fine nếu ∈ M_sp, M_fine nếu không
    M_pixel_applied = gt.copy()                              # base = M_fine = gt
    M_pixel_applied[M_sp > 0.5] = 1.0 - gt[M_sp > 0.5]    # flip pixel trong M_sp

    # ── Eq. 11: m_t = τ·M_obj_t + (1−τ)·M_pixel_applied_t ────────────
    tau = (np.random.rand(*gt.shape) < beta_b).astype(np.float32)
    m_t = tau * M_obj + (1 - tau) * M_pixel_applied

    return (m_t >= 0.5).astype(np.float32), M_obj, M_bnd, M_unc, M_sp

# ─── VISUALIZE ───────────────────────────────────────────────────────────────
STEPS = list(range(T_MAX))   # t = 0, 1, 2, 3, 4, 5
n_cols = 7   # coarse | overlay | diff | M_obj | M_bnd | M_unc | GT
fig, axes = plt.subplots(len(STEPS), n_cols, figsize=(26, 3.5 * len(STEPS)))
fig.patch.set_facecolor('#0d1117')

col_titles = ['m_t (coarse)', 'Overlay', 'FP/FN', 'M_obj_t', 'M_bnd_t', 'M_unc_t', 'GT']
for c, ttl in enumerate(col_titles):
    axes[0, c].set_title(ttl, color='#58a6ff', fontsize=10, fontweight='bold')

for i, t in enumerate(STEPS):
    coarse, M_obj, M_bnd, M_unc, M_sp = q_sample_vis(t)
    beta_b = betas_cumprod[t]

    # Overlay RGB + coarse (đỏ = foreground trong coarse)
    overlay = img_rgb.copy()
    overlay[coarse == 1] = (overlay[coarse == 1] * 0.55 + np.array([220, 50, 50]) * 0.45).clip(0, 255)

    # Diff map: TP=trắng, FP=đỏ, FN=xanh
    diff = np.zeros((*gt.shape, 3), dtype=np.float32)
    diff[(coarse == 1) & (gt == 1)] = [1.0, 1.0, 1.0]   # TP
    diff[(coarse == 1) & (gt == 0)] = [1.0, 0.0, 0.0]   # FP
    diff[(coarse == 0) & (gt == 1)] = [0.0, 0.5, 1.0]   # FN

    row_label = f"t={t}  β̄={beta_b:.2f}  n_bnd={t}  n_unc={T_MAX-t}"
    axes[i, 0].set_ylabel(row_label, color='#e6edf3', fontsize=8, rotation=0,
                          labelpad=110, va='center')

    for c, img_data in enumerate([coarse, overlay/255.0, diff, M_obj, M_bnd, M_unc, gt]):
        cmap = 'gray' if c in [0, 3, 4, 5, 6] else None
        axes[i, c].imshow(img_data, cmap=cmap, vmin=0, vmax=1)
        axes[i, c].axis('off')

plt.tight_layout(pad=1.5)
out = f"/home/ubuntu/vy/Denoiser/SegRefiner/train_coarse_vis_{NAME}.png"
plt.savefig(out, facecolor='#0d1117', dpi=100, bbox_inches='tight')
print(f"✅ Saved: {out}")
print(f"   Columns: {col_titles}")
print(f"   betas_cumprod = {betas_cumprod}")
print(f"   Eq.8 n_bnd=t (t=0→n=0, t=5→n=5 iterations)")
print(f"   Eq.6 n_unc=T-t (t=0→n=6, t=5→n=1 iterations)")
