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
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from skimage import measure
from sklearn.mixture import GaussianMixture
from mmdet.datasets.pipelines.loading import modify_boundary

# ─── CẤU HÌNH ────────────────────────────────────────────────────────────────
DATA_ROOT = '/home/ubuntu/vy/Denoiser/OpenEarthMap_wo_xBD'
CITY      = 'san_tome'
NAME      = 'san_tome_1'
T_MAX     = 6          # num_timesteps — betas_cumprod có 6 phần tử

# Beta schedule chuẩn bài báo: t=0 → 0.8 (sạch nhất), t=5 → 0.0 (bẩn nhất)
# β̄_t = linspace(0.8, 0.0, T)
betas_cumprod = np.linspace(0.8, 0.0, T_MAX)

# Ngưỡng tau_unc (Eq.6)
TAU_UNC = 0.5

# ─── ABLATION FLAGS ──────────────────────────────────────────────────────────
# Bật/tắt từng thành phần đóng góp vào m_t cuối cùng:
USE_M_OBJ      = True   # M_obj term trong Eq.11:  tau * M_obj
USE_M_APPLIED  = True   # M_pixel_applied term:     (1-tau) * M_pixel_applied
USE_MODIFY_BND = False   # Áp modify_boundary sau Eq.11 (nhiễu biên ngẫu nhiên)
# ─────────────────────────────────────────────────────────────────────────────

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

    # ── Eq. 7: M_obj_t — luôn tính để visualize (Eq.7) ──────────────────────
    labeled   = measure.label(gt > 0.5)
    instances = [(labeled == i) for i in range(1, labeled.max() + 1)]
    M_obj = gt.copy()   # default: giữ nguyên GT (không xóa object nào)
    if len(instances) > 0:
        unc_scores = [unc[inst].mean() for inst in instances]
        M_obj_new  = np.zeros_like(gt)
        kept_any   = False
        for inst_mask, u_k in zip(instances, unc_scores):
            if u_k <= beta_b:            # U_k^obj ≤ β_t → giữ lại
                M_obj_new[inst_mask] = 1.0
                kept_any = True
        if not kept_any:                 # fallback: giữ cái tốt nhất
            M_obj_new[instances[int(np.argmin(unc_scores))]] = 1.0
        M_obj = M_obj_new

    # ── Eq. 8: M_bnd_t — luôn tính để visualize ────────────────────────────
    n_bnd = t
    if n_bnd > 0:
        kernel_bnd = np.ones((3, 3), np.uint8)
        M_bnd = cv2.dilate((edge > 0.5).astype(np.uint8),
                           kernel_bnd, iterations=n_bnd).astype(np.float32)
    else:
        M_bnd = (edge > 0.5).astype(np.float32)

    # ── Eq. 6: M_unc_t — luôn tính để visualize ────────────────────────────
    unc_binary = (unc > TAU_UNC).astype(np.uint8)
    n_erode = T_MAX - t
    if n_erode > 0 and unc_binary.sum() > 0:
        kernel_unc = np.ones((3, 3), np.uint8)
        M_unc = cv2.erode(unc_binary, kernel_unc,
                          iterations=n_erode).astype(np.float32)
    else:
        M_unc = unc_binary.astype(np.float32)

    # ── Eq. 9-10: M_sp, M_pixel_applied — luôn tính ─────────────────────────
    M_sp            = ((M_bnd > 0.5) | (M_unc > 0.5)).astype(np.float32)
    M_pixel_applied = (M_sp * gt).astype(np.float32)

    # ── Eq. 11: m_t — gate theo ablation flags ───────────────────────────────
    tau = (np.random.rand(*gt.shape) < beta_b).astype(np.float32)
    m_obj_term     = M_obj     if USE_M_OBJ     else np.zeros_like(gt)
    m_applied_term = M_pixel_applied if USE_M_APPLIED else np.zeros_like(gt)
    m_t = tau * m_obj_term + (1 - tau) * m_applied_term

    # ── modify_boundary — scale theo timestep, gate theo USE_MODIFY_BND ──────
    # t=0 (sạch, β=0.8): nhẹ  → iou_target=0.90, rates=0.05
    # t=5 (bẩn, β=0.0): mạnh → iou_target=0.70, rates=0.20
    noise_level = t / max(T_MAX - 1, 1)               # 0.0 … 1.0
    mb_regional = 0.05 + 0.15 * noise_level
    mb_sample   = 0.05 + 0.15 * noise_level
    mb_iou      = 0.90 - 0.20 * noise_level
    m_t_before_mb = (m_t >= 0.5).astype(np.float32)   # Eq.11 output trước modify_bnd
    m_t_uint8 = (m_t >= 0.5).astype(np.uint8) * 255
    if USE_MODIFY_BND and m_t_uint8.sum() > 0:
        m_t_mb = modify_boundary(m_t_uint8,
                                 regional_sample_rate=mb_regional,
                                 sample_rate=mb_sample,
                                 iou_target=mb_iou)
    else:
        m_t_mb = (m_t_uint8 / 255).astype(np.uint8)   # không áp / mask rỗng

    M_pixel_applied_vis = M_pixel_applied.astype(np.float32)
    return (m_t_mb.astype(np.float32), M_obj, M_pixel_applied_vis,
            m_t_before_mb, M_sp, mb_regional, mb_iou)

# ─── VISUALIZE ───────────────────────────────────────────────────────────────
STEPS = list(range(T_MAX))   # t = 0, 1, 2, 3, 4, 5
n_cols = 7   # m_t_final | overlay | error | M_obj | M_pixel_applied | m_t_pre_mb | GT
fig, axes = plt.subplots(len(STEPS), n_cols, figsize=(28, 4.0 * len(STEPS)))
fig.patch.set_facecolor('#0d1117')
fig.suptitle(
    f"Q-Sample  |  {NAME}  |  T={T_MAX}  |  "
    f"M_obj={USE_M_OBJ}  M_applied={USE_M_APPLIED}  modify_bnd={USE_MODIFY_BND}",
    color='#e6edf3', fontsize=13, fontweight='bold', y=1.01
)

col_titles = [
    '① m_t  (final)\nEq.11 → modify_bnd',
    '② RGB overlay\nred = noisy foreground',
    '③ Error map\nwhite=TP · red=FP · blue=FN',
    '④ M_obj_t\nEq.7: object dropout',
    '⑤ M_pixel_applied\nEq.10: (M_bnd∪M_unc) ∩ GT',
    '⑥ m_t  before modify_bnd\nraw Eq.11 output',
    '⑦ GT mask\nground truth',
]
for c, ttl in enumerate(col_titles):
    axes[0, c].set_title(ttl, color='#58a6ff', fontsize=9, fontweight='bold',
                         linespacing=1.5)

for i, t in enumerate(STEPS):
    coarse, M_obj, M_pixel_applied, m_t_before, M_sp, mb_regional, mb_iou = q_sample_vis(t)
    beta_b = betas_cumprod[t]

    # Overlay RGB + coarse (đỏ = foreground trong coarse)
    overlay = img_rgb.copy()
    overlay[coarse == 1] = (overlay[coarse == 1] * 0.55 + np.array([220, 50, 50]) * 0.45).clip(0, 255)

    # Diff map: TP=trắng, FP=đỏ, FN=xanh
    diff = np.zeros((*gt.shape, 3), dtype=np.float32)
    diff[(coarse == 1) & (gt == 1)] = [1.0, 1.0, 1.0]   # TP
    diff[(coarse == 1) & (gt == 0)] = [1.0, 0.0, 0.0]   # FP
    diff[(coarse == 0) & (gt == 1)] = [0.0, 0.5, 1.0]   # FN

    row_label = (f"t={t}  β̄={beta_b:.2f}\n"
                 f"mb_rate={mb_regional:.2f}  iou_tgt={mb_iou:.2f}")
    axes[i, 0].set_ylabel(row_label, color='#e6edf3', fontsize=7.5, rotation=0,
                          labelpad=110, va='center')

    for c, img_data in enumerate([coarse, overlay/255.0, diff, M_obj, M_pixel_applied, m_t_before, gt]):
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
