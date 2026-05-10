"""
Visualization của Coarse Map — sync với q_sample thực tế trong training.

Pipeline (giống segrefiner_semantic.py::q_sample):
  Eq. 1 : unc_map  = H(GMM(x_rgb)) ∈ [0,1]   (precomputed .npy)
  Eq. 6 : M_unc_t  = Erode(1[unc > 0.5], n=T−t)
  Eq. 7 : M_obj_t  = giữ C_k nếu U_k^obj <= β_t, xóa nếu > β_t
  Eq. 9 : M_applied = M_unc_t ∩ GT
  Eq. 11: m_t       = τ·M_obj + (1−τ)·M_applied,  τ ~ Bernoulli(β_t)
          → modify_boundary(m_t, params scale theo t)

Không có Canny/M_bnd — thành phần này không tồn tại trong q_sample training.
"""
# [ignoring loop detection]
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import torch
import torch.nn.functional as F
from skimage import measure
from sklearn.mixture import GaussianMixture
from mmdet.datasets.pipelines.loading import modify_boundary

# ─── CẤU HÌNH ────────────────────────────────────────────────────────────────
DATA_ROOT = '/home/ubuntu/vy/Denoiser/OpenEarthMap_wo_xBD'
CITY      = 'tokyo'
NAME      = 'tokyo_2'
T_MAX     = 6          # num_timesteps

# β̄_t — dùng cho tau mixing (Bernoulli): tỉ lệ lấy M_obj vs M_applied
# stop=0.15 thay vì 0.0 → t=5 vẫn lấy 15% M_obj
betas_cumprod = np.linspace(0.3, 0.9, T_MAX)

# obj_thresholds: tính động từ unc_scores của ảnh (bên dưới sau khi load GT)

# Ngưỡng uncertainty pixel-level (Eq.6)
# Train chuẩn: 0.5 — nhưng unc_mean=0.24 nên 0.5 cho M_unc rất thưa
# Hạ xuống 0.3 để giữ nhiều vùng uncertain hơn
TAU_UNC = 0.5   # chuẩn training

# ─── ABLATION FLAGS (sync với exp4_all) ──────────────────────────────────────
USE_M_OBJ      = True   # M_obj term (Eq.7)
USE_M_UNC      = True   # M_applied term (Eq.9-10)
USE_MODIFY_BND = True   # modify_boundary sau Eq.11 (đã sync với train)
# ─────────────────────────────────────────────────────────────────────────────

# ─── ĐỌC DỮ LIỆU ─────────────────────────────────────────────────────────────
img_bgr = cv2.imread(f"{DATA_ROOT}/{CITY}/images/{NAME}.tif")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gt      = (cv2.imread(f"{DATA_ROOT}/{CITY}/labels/{NAME}.tif", 0) == 1).astype(np.float32)

# ── Eq. 1: unc_map ───────────────────────────────────────────────────────────
unc_path = f"{DATA_ROOT}/{CITY}/uncertainty/{NAME}.npy"
if os.path.exists(unc_path):
    unc = np.load(unc_path)
    print(f"DEBUG: Loaded precomputed uncertainty map")
else:
    pix = (img_rgb.astype(np.float32) / 255.0).reshape(-1, 3)
    idx = np.random.choice(len(pix), min(20_000, len(pix)), replace=False)
    samples = pix[idx]
    best_bic = np.inf
    best_gmm = None
    for k in [2, 3, 4, 5]:
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42).fit(samples)
        bic = gmm.bic(samples)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
    best_k = best_gmm.n_components
    proba  = best_gmm.predict_proba(pix)
    entropy = -np.sum(proba * np.log(proba + 1e-8), axis=1) / np.log(2)
    unc = entropy.reshape(img_rgb.shape[:2]).astype(np.float32)
    unc = np.clip(unc, 0.0, 1.0)
    print(f"DEBUG: Computed GMM on-the-fly with best_k={best_k}")

print(f"DEBUG: unc_map  max={unc.max():.4f}  mean={unc.mean():.4f}")

# ─── PRECOMPUTE: instances + unc_scores + dynamic obj_thresholds ──────────────
_labeled    = measure.label(gt > 0.5)
instances   = [(_labeled == i) for i in range(1, _labeled.max() + 1)]
if len(instances) > 0:
    unc_scores = [unc[inst].mean() for inst in instances]
    score_max  = max(unc_scores)
    score_min  = min(unc_scores)
else:
    unc_scores = []
    score_max  = score_min = 0.0

# Cbrt schedule: lõm mạnh nhất → drop cực nhanh t=0→1, rất chậm ở t=3,4,5
_t    = np.arange(T_MAX)
alpha = (_t / (T_MAX - 1)) ** (1/3)              # [0 → 1], lõm cực mạnh
obj_thresholds_img = score_max - alpha * (score_max - score_max / 2.5)
# t=0→score_max (fast drop) → t=5→score_max/8 (slow, clustered)
print(f"DEBUG: score_max={score_max:.3f}  score_min={score_min:.3f}")
print(f"DEBUG: obj_thresholds_img = {np.round(obj_thresholds_img, 3)}")

# ─── HÀM Q_SAMPLE (mirror segrefiner_semantic.py::q_sample) ──────────────────
def q_sample_vis(t):
    beta_b  = betas_cumprod[t]    # dùng cho tau mixing (Bernoulli)

    # ── Eq. 7: M_obj_t — dynamic threshold theo unc_scores của ảnh ───────────
    obj_thr = obj_thresholds_img[t]  # linspace(score_max, score_min, T_MAX)
    M_obj = gt.copy()
    if len(instances) > 0:
        M_obj_new  = np.zeros_like(gt)
        kept_any   = False
        for inst_mask, u_k in zip(instances, unc_scores):
            if u_k <= obj_thr:
                M_obj_new[inst_mask] = 1.0
                kept_any = True
        if not kept_any:               # fallback: giữ 1 cái chắc chắn nhất
            M_obj_new[instances[int(np.argmin(unc_scores))]] = 1.0
        M_obj = M_obj_new

    # ── Eq. 6: M_unc_t — DILATION tăng dần ───────────────────────────────────
    unc_binary = torch.tensor((unc > TAU_UNC).astype(np.float32)).unsqueeze(0).unsqueeze(0)
    n_dilate = T_MAX - t  # t=0→6 lần (nhiều nhiễu), t=5→1 lần (sạch)
    if n_dilate > 0 and unc_binary.sum() > 0:
        m_unc_t = unc_binary
        for _ in range(n_dilate):
            m_unc_t = F.max_pool2d(m_unc_t, kernel_size=3, stride=1, padding=1)
        M_unc = m_unc_t.squeeze().numpy()
    else:
        M_unc = unc_binary.squeeze().numpy()

    # ── Eq. 9-10: M_applied = M_unc ∩ GT ─────────────────────────────────────
    M_applied = ((M_unc > 0.5) * gt).astype(np.float32)

    # ── Eq. 11: m_t = τ·M_obj + (1−τ)·M_applied ─────────────────────────────
    if not USE_M_OBJ and not USE_M_UNC:
        # Mirror training: fallback về GT (không apply noise object/unc)
        m_t = gt.copy()
    else:
        m_obj_term     = M_obj     if USE_M_OBJ else np.zeros_like(gt)
        m_applied_term = M_applied if USE_M_UNC else np.zeros_like(gt)
        tau = (np.random.rand(*gt.shape) < beta_b).astype(np.float32)
        # ĐẢO VỊ TRÍ: tau giờ là xác suất lấy M_applied
        m_t = tau * m_applied_term + (1 - tau) * m_obj_term

    # Sin schedule cho bnd: tăng nhanh ở t thấp, chậm ở t cao
    # + giảm nhiễu tổng thể (range nhỏ hơn)
    # ── modify_boundary — scale theo t (linear schedule chuẩn training) ─────
    noise_level   = t / max(T_MAX - 1, 1)
    mb_regional   = 0.05 * noise_level
    mb_sample     = 0.5 
    mb_iou        = 1 - 0.01 * noise_level

    m_t_before_mb = (m_t >= 0.5).astype(np.float32)
    m_t_uint8     = (m_t >= 0.5).astype(np.uint8) * 255
    if USE_MODIFY_BND and m_t_uint8.sum() > 0:
        m_t_mb = modify_boundary(m_t_uint8,
                                 regional_sample_rate=mb_regional,
                                 sample_rate=mb_sample,
                                 iou_target=mb_iou)
    else:
        m_t_mb = (m_t_uint8 / 255).astype(np.uint8)

    return m_t_mb.astype(np.float32), M_obj, M_applied, M_unc, mb_regional, mb_iou


# ─── VISUALIZE ───────────────────────────────────────────────────────────────
STEPS  = list(range(T_MAX))
n_cols = 7   # m_t_final | overlay | error | M_obj | M_applied | m_t_pre_mb | GT | unc
fig, axes = plt.subplots(len(STEPS), n_cols, figsize=(28, 4.0 * len(STEPS)))
fig.patch.set_facecolor('#0d1117')
fig.suptitle(
    f"Q-Sample  |  {NAME}  |  T={T_MAX}  |  "
    f"M_obj={USE_M_OBJ}  M_unc={USE_M_UNC}  modify_bnd={USE_MODIFY_BND}",
    color='#e6edf3', fontsize=13, fontweight='bold', y=1.01)

col_titles = [
    '① m_t (final)\nEq.11 → mod_bnd',
    '② Original RGB',
    '③ Error map',
    '④ M_obj_t\nEq.7: obj dropout',
    '⑤ M_applied\nEq.9: M_unc ∩ GT',
    '⑥ M_unc_t\nEq.6: Dilation',
    '⑦ GT mask',
]
for c, ttl in enumerate(col_titles):
    axes[0, c].set_title(ttl, color='#58a6ff', fontsize=9, fontweight='bold')

for i, t in enumerate(STEPS):
    coarse, M_obj, M_applied, M_unc_vis, mb_regional, mb_iou = q_sample_vis(t)

    # Column 2: RGB gốc
    overlay = img_rgb.copy()

    diff = np.zeros((*gt.shape, 3), dtype=np.float32)
    diff[(coarse == 1) & (gt == 1)] = [1.0, 1.0, 1.0]   # TP: trắng
    diff[(coarse == 1) & (gt == 0)] = [1.0, 0.0, 0.0]   # FP: đỏ
    diff[(coarse == 0) & (gt == 1)] = [0.0, 0.5, 1.0]   # FN: xanh

    axes[i, 0].set_ylabel(
        f"t={t}  β̄={betas_cumprod[t]:.2f}\n"
        f"iou_tgt={mb_iou:.2f}  rate={mb_regional:.2f}",
        color='#e6edf3', fontsize=7, rotation=0, labelpad=55, va='center')

    img_list = [coarse, overlay / 255.0, diff, M_obj, M_applied, M_unc_vis, gt]
    for c, img_data in enumerate(img_list):
        cmap = 'gray' if c in [0, 3, 4, 5, 6] else None
        axes[i, c].imshow(img_data, cmap=cmap, vmin=0, vmax=1)
        axes[i, c].axis('off')

plt.tight_layout()
out = f"/home/ubuntu/vy/Denoiser/SegRefiner/train_coarse_vis_{NAME}.png"
plt.savefig(out, facecolor='#0d1117', dpi=100, bbox_inches='tight')
print(f"✅ Saved: {out}")
