"""
Precompute uncertainty maps + Canny edge maps — Y CHANG BÀI BÁO.

Eq. 1 : M_unc_coarse  = H(W(x_rgb)) / σ  ∈ [0,1]
         W = weak classifier (GMM 2 components trên RGB)
         H = entropy của posterior probabilities
         / σ  = chuẩn hóa về [0,1] (với GMM 2 class, H_max = log2(2) = 1 bit)

Eq. 5 : M_bnd_coarse  = Canny(x_rgb)
         RAW Canny — không dilate thêm ở bước precompute.
         Việc Dilate(n=t) sẽ được thực hiện ON-THE-FLY trong q_sample (Eq. 8).
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
RECOMPUTE_CANNY       = False   # Tắt vì Canny đã chuẩn Eq.5 rồi

# Thông số Canny (Eq. 5)
CANNY_LOW     = 50
CANNY_HIGH    = 150
# Không lọc connected component — giữ nguyên tất cả biên từ Canny
# (dilation sẽ làm sau, tại q_sample, với n_iter = t)

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


def compute_canny_raw(img_rgb):
    """
    Eq. 5: M_bnd_coarse = Canny(x_rgb)

    RAW Canny — không dilate thêm ở đây.
    Eq. 8 sẽ Dilate(M_bnd, n_iter=t) on-the-fly trong q_sample.
    Output: float32 binary {0.0, 1.0}.
    """
    img_blur = cv2.GaussianBlur(img_rgb, (5, 5), 0)
    gray     = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    edge_raw = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    return (edge_raw.astype(np.float32) / 255.0)   # {0.0, 1.0}


def process_one_image(args):
    img_path, unc_path, edge_path = args
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return False
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Eq. 1: GMM uncertainty
    if RECOMPUTE_UNCERTAINTY or not osp.exists(unc_path):
        unc = compute_gmm_uncertainty(img_rgb)
        np.save(unc_path, unc)

    # Eq. 5: Raw Canny edge
    if RECOMPUTE_CANNY or not osp.exists(edge_path):
        edge = compute_canny_raw(img_rgb)
        np.save(edge_path, edge)

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
        edge_path = osp.join(DATA_ROOT, city, 'edges',       basename + '.npy')

        os.makedirs(osp.dirname(unc_path),  exist_ok=True)
        os.makedirs(osp.dirname(edge_path), exist_ok=True)

        need_unc  = RECOMPUTE_UNCERTAINTY or not osp.exists(unc_path)
        need_edge = RECOMPUTE_CANNY       or not osp.exists(edge_path)

        if need_unc or need_edge:
            pending.append((img_path, unc_path, edge_path))

    if pending:
        print(f"Processing {len(pending)} images  (GMM={RECOMPUTE_UNCERTAINTY}, Canny={RECOMPUTE_CANNY})")
        print("  Eq.1: uncertainty = H(GMM) / log2  ∈ [0,1]")
        print("  Eq.5: edge = raw Canny (no dilation — Eq.8 dilates on-the-fly)")
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            list(tqdm(executor.map(process_one_image, pending), total=len(pending)))
    else:
        print("Tất cả file đã tồn tại, không cần xử lý.")

    print("✅ Completed!")


if __name__ == '__main__':
    main()
