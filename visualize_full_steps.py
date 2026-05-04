"""
visualize_full_steps.py
Mô phỏng quá trình tạo Coarse Mask từ t=0 đến t=12.
"""
import cv2, numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from sklearn.mixture import GaussianMixture

# ── Cấu hình ──
CITY = 'melbourne'
NAME = 'melbourne_48'
DATA_ROOT = '/home/ubuntu/vy/Denoiser/OpenEarthMap_wo_xBD'
T_MAX = 12
STEPS = [0, 2, 4, 6, 8, 10, 12]

# ── Load ──
img = cv2.cvtColor(cv2.imread(f"{DATA_ROOT}/{CITY}/images/{NAME}.tif"), cv2.COLOR_BGR2RGB)
gt  = (cv2.imread(f"{DATA_ROOT}/{CITY}/labels/{NAME}.tif", 0) == 1).astype(np.float32)

# ── Prep Maps ──
print("Computing maps...")
# Extract edges from GT (The correct way for Boundary Noise)
kernel_edge = np.ones((3,3), np.uint8)
gt_dilated = cv2.dilate(gt, kernel_edge)
gt_eroded = cv2.erode(gt, kernel_edge)
gt_edge = (gt_dilated - gt_eroded)

# GMM Uncertainty
pix = (img.astype(np.float32) / 255.0).reshape(-1, 3)
idx = np.random.choice(len(pix), 50_000, replace=False)
gmm = GaussianMixture(2, covariance_type='full', random_state=42).fit(pix[idx])
proba = gmm.predict_proba(pix)
unc = (-np.sum(proba * np.log(proba + 1e-8), axis=1) / np.log(2)).reshape(img.shape[:2])

def generate_coarse(t):
    if t == 0: return gt.copy()
    
    # 1. Object noise (Simulation: threshold decreases as t increases)
    obj_thresh = 1.0 - (t / T_MAX) * 0.7 
    labeled, n = ndi.label(gt.astype(np.uint8))
    m_obj = gt.copy()
    for i in range(1, n+1):
        m = (labeled == i)
        if unc[m].mean() > obj_thresh: 
            m_obj[m] = 0.0 # Remove object
            
    # 2. Boundary noise (Kernel size proportional to t)
    k_size = t * 2 + 1
    m_bnd = cv2.dilate(gt_edge, np.ones((k_size, k_size), np.uint8))
    
    # 3. Uncertainty noise (More noise remains as t increases)
    bin_unc = (unc > 0.4).astype(np.float32)
    m_unc = cv2.erode(bin_unc, np.ones((3,3), np.uint8), iterations=(T_MAX - t))
    
    # Combine Noise Sources
    noise_mask = np.logical_or(np.logical_or(m_bnd > 0.5, m_unc > 0.5), (gt != m_obj))
    
    res = gt.copy()
    res[noise_mask] = 1 - res[noise_mask]
    return res

# ── Plot ──
fig, axes = plt.subplots(len(STEPS), 3, figsize=(18, 4 * len(STEPS)))
fig.patch.set_facecolor('#0d1117')

for i, t in enumerate(STEPS):
    coarse = generate_coarse(t)
    # RGB + Coarse overlay
    overlay = img.copy()
    overlay[coarse == 1] = overlay[coarse == 1] * 0.5 + np.array([255, 0, 0]) * 0.5 # Red tint
    
    # Diff map
    diff = np.zeros((*gt.shape, 3))
    diff[np.logical_and(coarse==1, gt==1)] = [1, 1, 1] # White: Correct
    diff[np.logical_and(coarse==1, gt==0)] = [1, 0, 0] # Red: False Positive
    diff[np.logical_and(coarse==0, gt==1)] = [0, 0.5, 1] # Blue: False Negative

    axes[i, 0].imshow(coarse, cmap='gray'); axes[i, 0].set_title(f"Coarse Mask (t={t})", color='white')
    axes[i, 1].imshow(overlay.astype(np.uint8)); axes[i, 1].set_title(f"Overlay on RGB (t={t})", color='white')
    axes[i, 2].imshow(diff); axes[i, 2].set_title(f"Errors (Red=Add, Blue=Remove)", color='white')
    for ax in axes[i]: ax.axis('off')

plt.tight_layout()
plt.savefig("/home/ubuntu/vy/Denoiser/SegRefiner/full_steps_visualized.png", facecolor='#0d1117')
print("Done: full_steps_visualized.png")
