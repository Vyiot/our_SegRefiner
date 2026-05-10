"""
Precompute uncertainty maps — Sync với train code hiện tại.

Eq. 1 : M_unc_coarse  = H(W(x_rgb)) / σ  ∈ [0,1]
         W = weak classifier (GMM n components trên RGB, tìm n bằng BIC)
         H = entropy của posterior probabilities
         / σ  = chuẩn hóa về [0,1] (σ = log(K))
"""
import os
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from concurrent.futures import ProcessPoolExecutor

# ─── CẤU HÌNH ────────────────────────────────────────────────────────────────
DATA_ROOT  = '/home/ubuntu/vy/Denoiser/OpenEarthMap_wo_xBD'
SPLITS     = ['train.txt']

# Chạy lại hay bỏ qua nếu file đã tồn tại
RECOMPUTE_UNCERTAINTY = True    # Bật để tính lại GMM với K tối ưu

NUM_WORKERS = 8
# ──────────────────────────────────────────────────────────────────────────────


def get_city(data_root, basename):
    """Tìm tên thành phố từ basename bằng cách match với thư mục con."""
    parts = basename.split('_')
    for i in range(len(parts), 0, -1):
        candidate = '_'.join(parts[:i])
        if osp.isdir(osp.join(data_root, candidate)):
            return candidate
    return None


def compute_gmm_uncertainty(img_rgb):
    """
    Eq. 1: M_unc_coarse = H(W(x_rgb)) / σ

    W = GMM với n_components tối ưu tìm bằng BIC (Bayesian Information Criterion).
    H = entropy của posterior = -Σ p·log(p).
    σ = log(K)  [max entropy của GMM K class]
    → output ∈ [0, 1].
    """
    H, W, C = img_rgb.shape
    pixels   = img_rgb.astype(np.float32).reshape(-1, C) / 255.0

    # Subsample để tăng tốc tìm K và Fit
    idx = np.random.choice(pixels.shape[0], min(50_000, pixels.shape[0]), replace=False)
    sub_pixels = pixels[idx]

    # ── Tìm K tốt nhất bằng BIC (2 đến 4) ────────────────────────────────────
    best_k = 2
    best_bic = np.inf
    best_gmm = None
    
    for k in range(2, 5):
        gmm = GaussianMixture(
            n_components=k, covariance_type='full',
            max_iter=50, random_state=42
        )
        gmm.fit(sub_pixels)
        bic = gmm.bic(sub_pixels)
        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_gmm = gmm

    # Tính posterior cho tất cả pixels bằng GMM tốt nhất
    proba   = best_gmm.predict_proba(pixels)                              # (N, best_k)
    entropy = -np.sum(proba * np.log(proba + 1e-8), axis=1)              # (N,)

    # Chuẩn hóa: chia cho log(best_k) để ∈ [0, 1]
    unc = (entropy / np.log(best_k)).reshape(H, W).astype(np.float32)
    return np.clip(unc, 0.0, 1.0)




def process_one_image(args):
    img_path, unc_path = args
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return False
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Eq. 1: GMM uncertainty
    if RECOMPUTE_UNCERTAINTY or not osp.exists(unc_path):
        unc = compute_gmm_uncertainty(img_rgb)
        np.save(unc_path, unc)

    return True


def main():
    all_imgs = set()
    for split in SPLITS:
        split_file = osp.join(DATA_ROOT, split)
        if not osp.exists(split_file):
            continue
        with open(split_file) as f:
            for line in f:
                name = line.strip()
                if name:
                    all_imgs.add(name)

    all_imgs = sorted(all_imgs)
    pending  = []

    for img_name in all_imgs:
        basename  = osp.splitext(osp.basename(img_name))[0]
        city      = get_city(DATA_ROOT, basename)
        if not city:
            continue
        img_path  = osp.join(DATA_ROOT, city, 'images',      basename + '.tif')
        unc_path  = osp.join(DATA_ROOT, city, 'uncertainty', basename + '.npy')

        os.makedirs(osp.dirname(unc_path),  exist_ok=True)

        if RECOMPUTE_UNCERTAINTY or not osp.exists(unc_path):
            pending.append((img_path, unc_path))

    if pending:
        print(f"Processing {len(pending)} images  (GMM_recompute={RECOMPUTE_UNCERTAINTY})")
        print("  Eq.1: uncertainty = H(GMM) / log(K)  ∈ [0,1]")
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            list(tqdm(executor.map(process_one_image, pending), total=len(pending)))
    else:
        print("Tất cả file đã tồn tại, không cần xử lý.")

    print("✅ Completed!")


if __name__ == '__main__':
    main()
