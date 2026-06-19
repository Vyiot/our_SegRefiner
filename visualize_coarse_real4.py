"""
Visualization của Coarse Map ÁP DỤNG ĐÚNG CÁC CÔNG THỨC BÀI BÁO (Eq. 1–11).

Eq. 1 : M_unc_coarse  = H(W(x_rgb)) / σ  ∈ [0,1]   (GMM entropy, normalized)
Eq. 3 : U_k^obj       = (1/|C_k|) * Σ M_coarse_unc(i,j)  for (i,j) ∈ C_k
Eq. 4 : M_coarse^obj   = U_k^obj nếu (i,j) ∈ C_k, 0 nếu background
Eq. 5 : M_bnd_coarse  = modify_boundary(M_fine, t)   (BOUNDARY NOISE)
Eq. 6 : M_unc_t       = Erode( 1[M_unc > τ_unc], n_iter=T−t )
Eq. 7 : M_obj_t(i,j)  = 0 nếu (i,j)∈C_k AND U_k^obj > β̄_t, else M_fine(i,j)
Eq. 8 : M_bnd_t       = Dilate( M_bnd_coarse, n_iter=t )
Eq. 9 : I_fused       = M_unc_t ∪ M_bnd_t
Eq.10 : M_pixel_t     = 1_{fused=1}·(1−M_fine) + 1_{fused=0}·M_fine
Eq.11 : m_t           = τ^{i,j}·M_obj_t + (1−τ^{i,j})·M_pixel_t, τ~Bernoulli(β̄_t)
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
CITY      = 'paris'
NAME      = 'paris_1'
T_MAX     = 6          # num_timesteps — betas_cumprod có 6 phần tử

# Beta schedule chuẩn bài báo: t=0 → 0.8 (sạch nhất), t=5 → 0.0 (bẩn nhất)
# β̄_t = linspace(0.8, 0.0, T)
betas_cumprod = np.linspace(0.8, 0.0, T_MAX)

# Ngưỡng tau_unc (Eq.6)
TAU_UNC = 0.3

# ─── ABLATION FLAGS ──────────────────────────────────────────────────────────
# Bật/tắt từng thành phần đóng góp vào m_t cuối cùng:
USE_M_OBJ      = True   # M_obj term trong Eq.11:  tau * M_obj
USE_M_APPLIED  = True   # M_pixel_applied term:     (1-tau) * M_pixel_applied
USE_MODIFY_BND = True   # Áp modify_boundary sau Eq.11 (nhiễu biên ngẫu nhiên)
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

# ── Eq. 5: M_bnd_coarse = modify_boundary(M_fine, t) ────────────────────────
# M_bnd_coarse sẽ được tính trong q_sample_vis() vì phụ thuộc vào t

def my_modify_boundary(image, regional_sample_rate=0.1, sample_rate=0.1, move_rate=0.0, iou_target=0.8):
    # 1. Tạo coarse mask gốc
    coarse_mask = modify_boundary(image, regional_sample_rate, sample_rate, move_rate, iou_target)
    # 2. Trích xuất biên (boundary) của coarse mask đó
    coarse_uint8 = (coarse_mask * 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(coarse_uint8, kernel, iterations=1)
    eroded = cv2.erode(coarse_uint8, kernel, iterations=1)
    boundary = cv2.subtract(dilated, eroded)
    return (boundary > 127).astype(np.float32)

# ─── HÀM Q_SAMPLE ────────────────────────────────────────────────────────────
def q_sample_vis(t):
    """
    Áp dụng ĐÚNG Eq. 6-11 của bài báo.

    Lịch betas: betas_cumprod = linspace(0.8→0.0, T=6)
        t=0 → β̄=0.8  (sạch nhất — ít nhiễu)
        t=5 → β̄=0.0  (bẩn nhất — nhiều nhiễu)
    """
    beta_b = betas_cumprod[t]   # β̄_t

    # ── Eq. 3: U_k^obj = mean uncertainty of each connected component ────────
    labeled   = measure.label(gt > 0.5)
    instances = [(labeled == i) for i in range(1, labeled.max() + 1)]
    unc_scores = [unc[inst].mean() for inst in instances] if len(instances) > 0 else []

    # ── Eq. 4: M_coarse^obj(i,j) = U_k^obj nếu (i,j) ∈ C_k, 0 nếu bg ──────
    M_coarse_obj = np.zeros_like(gt)
    for inst_mask, u_k in zip(instances, unc_scores):
        M_coarse_obj[inst_mask] = u_k

    # ── Eq. 7: M_obj_t(i,j) = 0 nếu (i,j)∈C_k AND U_k > β̄_t ──────────────
    #           M_obj_t(i,j) = M_fine(i,j) otherwise
    M_obj = gt.copy()   # default: M_fine(i,j) cho mọi pixel (kể cả background=0)
    if len(instances) > 0:
        for inst_mask, u_k in zip(instances, unc_scores):
            if u_k > beta_b:             # U_k^obj > β̄_t → xóa (set = 0)
                M_obj[inst_mask] = 0.0
            # else: giữ nguyên M_fine(i,j) = gt(i,j) đã copy ở trên

    # ── Eq. 5: M_bnd_coarse = my_modify_boundary(M_fine, t) ─────────────────
    gt_uint8 = (gt * 255).astype(np.uint8)
    if gt_uint8.sum() > 0:
        M_bnd_coarse = my_modify_boundary(gt_uint8).astype(np.float32)
    else:
        M_bnd_coarse = np.zeros_like(gt)

    # ── Eq. 8: M_bnd_t = Dilate(M_bnd_coarse, n_iter=t) ────────────────────
    n_bnd = t
    if n_bnd > 0:
        kernel_bnd = np.ones((3, 3), np.uint8)
        M_bnd = cv2.dilate((M_bnd_coarse > 0.5).astype(np.uint8),
                           kernel_bnd, iterations=n_bnd).astype(np.float32)
    else:
        M_bnd = (M_bnd_coarse > 0.5).astype(np.float32)

    # ── Eq. 6: M_unc_t — luôn tính để visualize ────────────────────────────
    unc_binary = (unc > TAU_UNC).astype(np.uint8)
    n_erode = T_MAX - t
    if n_erode > 0 and unc_binary.sum() > 0:
        kernel_unc = np.ones((3, 3), np.uint8)
        M_unc = cv2.erode(unc_binary, kernel_unc,
                          iterations=n_erode).astype(np.float32)
    else:
        M_unc = unc_binary.astype(np.float32)

    # ── Eq. 9: I_fused = M_unc_t ∪ M_bnd_t ──────────────────────────────────
    I_fused = ((M_bnd > 0.5) | (M_unc > 0.5)).astype(np.float32)

    # ── Eq.10: M_pixel_t = 1_{fused=1}·(1−M_fine) + 1_{fused=0}·M_fine ─────
    M_pixel = I_fused * (1.0 - gt) + (1.0 - I_fused) * gt

    # ── Eq.11: m_t = τ·M_obj_t + (1−τ)·M_pixel_t, τ~Bernoulli(β̄_t) ────────
    tau = (np.random.rand(*gt.shape) < beta_b).astype(np.float32)
    m_obj_term   = M_obj   if USE_M_OBJ     else np.zeros_like(gt)
    m_pixel_term = M_pixel if USE_M_APPLIED else np.zeros_like(gt)
    m_t = tau * m_obj_term + (1 - tau) * m_pixel_term

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

    return (m_t_mb.astype(np.float32), M_coarse_obj, M_obj, M_bnd_coarse, M_bnd,
            M_unc, I_fused, M_pixel, m_t_before_mb, mb_regional, mb_iou)

# ─── VISUALIZE ───────────────────────────────────────────────────────────────
STEPS = list(range(T_MAX))   # t = 0, 1, 2, 3, 4, 5
n_cols = 9   # RGB | GT | M_unc | M_obj | M_bnd | I_fused | M_pixel | m_t | Error
fig, axes = plt.subplots(len(STEPS), n_cols, figsize=(40, 4.5 * len(STEPS)))
fig.patch.set_facecolor('#0d1117')
fig.suptitle(
    f"Q-Sample  |  {NAME}  |  T={T_MAX}  |  "
    f"M_obj={USE_M_OBJ}  M_pixel={USE_M_APPLIED}  modify_bnd={USE_MODIFY_BND}",
    color='#e6edf3', fontsize=16, fontweight='bold', y=1.01
)

col_titles = [
    '① RGB\noriginal',
    '② GT\nground truth',
    '③ M_unc_t\nEq.6 eroded uncertainty',
    '④ M_obj_t\nEq.7 object dropout',
    '⑤ M_bnd_t\nEq.8 dilated bnd',
    '⑥ I_fused\nEq.9 M_unc ∪ M_bnd',
    '⑦ M_pixel_t\nEq.10 flip/keep',
    '⑧ m_t\nEq.11 final',
    '⑨ Error map\nTP=W · FP=R · FN=B',
]
for c, ttl in enumerate(col_titles):
    axes[0, c].set_title(ttl, color='#58a6ff', fontsize=13, fontweight='bold',
                         linespacing=1.45)

for i, t in enumerate(STEPS):
    (coarse, M_coarse_obj, M_obj, M_bnd_coarse, M_bnd,
     M_unc, I_fused, M_pixel, m_t_before, mb_regional, mb_iou) = q_sample_vis(t)
    beta_b = betas_cumprod[t]

    # Diff map: TP=trắng (1,1,1), FP=đỏ (1,0,0), FN=xanh lam (0,0.5,1)
    diff = np.zeros((*gt.shape, 3), dtype=np.float32)
    diff[(coarse == 1) & (gt == 1)] = [1.0, 1.0, 1.0]   # TP
    diff[(coarse == 1) & (gt == 0)] = [1.0, 0.0, 0.0]   # FP
    diff[(coarse == 0) & (gt == 1)] = [0.0, 0.5, 1.0]   # FN

    # Tính phần trăm lỗi FP, FN so với tổng diện tích GT
    gt_sum = max(gt.sum(), 1.0)
    fp_pixels = ((coarse == 1) & (gt == 0)).sum()
    fn_pixels = ((coarse == 0) & (gt == 1)).sum()
    fp_percent = (fp_pixels / gt_sum) * 100
    fn_percent = (fn_pixels / gt_sum) * 100

    row_label = (f"t={t}  β̄={beta_b:.2f}\n"
                 f"n_bnd={t}  n_unc={T_MAX-t}\n"
                 f"FP={fp_percent:.1f}%\nFN={fn_percent:.1f}%")
    axes[i, 0].set_ylabel(row_label, color='#e6edf3', fontsize=9, rotation=0,
                          labelpad=95, va='center')

    img_list = [
        img_rgb / 255.0,       # ① RGB
        gt,                    # ② GT
        M_unc,                 # ③ M_unc_t (Eq.6 eroded uncertainty)
        M_obj,                 # ④ M_obj_t
        M_bnd,                 # ⑤ M_bnd_t
        I_fused,               # ⑥ I_fused
        M_pixel,               # ⑦ M_pixel_t
        coarse,                # ⑧ m_t final
        diff,                  # ⑨ Error map
    ]
    for c, img_data in enumerate(img_list):
        cmap = 'gray' if c in [1, 2, 3, 4, 5, 6, 7] else None
        axes[i, c].imshow(img_data, cmap=cmap, vmin=0, vmax=1)
        axes[i, c].axis('off')

    print(f"   t={t}: FP={fp_percent:.2f}% (pixels={fp_pixels}), FN={fn_percent:.2f}% (pixels={fn_pixels})")

plt.tight_layout(pad=1.5)
out = f"/home/ubuntu/vy/Denoiser/SegRefiner/train_coarse_vis_{NAME}_ver4.png"
plt.savefig(out, facecolor='#0d1117', dpi=100, bbox_inches='tight')
print(f"✅ Saved: {out}")
print(f"   Columns: {[t.split(chr(10))[0] for t in col_titles]}")
print(f"   betas_cumprod = {betas_cumprod}")
