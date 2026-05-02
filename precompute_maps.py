"""
Precompute uncertainty maps + Canny edge maps cho toàn bộ OEM dataset.

  Uncertainty: GMM + Entropy (đúng theo paper) - fit trên subsample pixels
  Canny:       Kornia GPU Canny (batch trên GPU)

Kết quả lưu vào:
  {data_root}/{city}/uncertainty/{basename}.npy   ← entropy ∈ [0, 1]
  {data_root}/{city}/edges/{basename}.npy         ← binary edges {0, 1}

Tốc độ ước tính:
  - GMM (subsample 50k): ~3-5s/ảnh trên CPU
  - Canny batch GPU: ~0.05s/ảnh
  - Tổng ~3000 ảnh: ~3-5 giờ (có thể dùng NUM_WORKERS để song song hóa)
"""
import os
import os.path as osp
import cv2
import numpy as np

# Tắt cảnh báo TIFF của OpenCV
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
import torch
import kornia
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from concurrent.futures import ProcessPoolExecutor, as_completed

# ─── Cấu hình ─────────────────────────────────────────────────────────────────
DATA_ROOT       = '/home/ubuntu/vy/Denoiser/OpenEarthMap_wo_xBD'
SPLITS          = ['train.txt']
CANNY_LOW       = 0.2           # Tương đương 50/255 (loại bỏ nhiễu texture)
CANNY_HIGH      = 0.6           # Tương đương 150/255
CANNY_BATCH     = 32            # Số ảnh xử lý cùng lúc qua GPU Canny
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'

# GMM config
GMM_N_COMPONENTS = 2            # Số cụm: foreground / background
GMM_MAX_ITER     = 100          # Số vòng lặp EM tối đa
GMM_N_SAMPLE     = 50_000       # Số pixel subsample để fit GMM (tốc độ/chất lượng)
GMM_RANDOM_SEED  = 42

NUM_WORKERS      = 1            # Chạy tuần tự (không parallel) để đỡ tốn tài nguyên
FORCE_RECOMPUTE  = True         # Xóa và tính lại từ đầu
# ──────────────────────────────────────────────────────────────────────────────


def get_city(data_root, basename):
    parts = basename.split('_')
    for i in range(len(parts), 0, -1):
        candidate = '_'.join(parts[:i])
        if osp.isdir(osp.join(data_root, candidate)):
            return candidate
    return None


# ── GMM Uncertainty (theo paper) ─────────────────────────────────────────────
def compute_gmm_uncertainty(img_rgb: np.ndarray,
                             n_components: int = GMM_N_COMPONENTS,
                             n_sample: int     = GMM_N_SAMPLE,
                             max_iter: int     = GMM_MAX_ITER) -> np.ndarray:
    """
    Tính U_score = Entropy của posterior GMM trên pixel RGB.

    Trick tăng tốc: fit GMM trên n_sample pixel ngẫu nhiên,
    sau đó predict_proba trên toàn bộ ảnh.
    Kết quả gần như tương đương fit trên toàn ảnh.

    Args:
        img_rgb: (H, W, 3) uint8 [0-255]

    Returns:
        uncertainty: (H, W) float32 [0, 1]
                     Cao = pixel màu nhập nhằng giữa các cụm
    """
    H, W, C = img_rgb.shape
    img_norm = img_rgb.astype(np.float32) / 255.0
    pixels   = img_norm.reshape(-1, C)          # (H*W, 3)

    # ── Subsample để fit nhanh ──
    N = pixels.shape[0]
    if N > n_sample:
        rng = np.random.default_rng(GMM_RANDOM_SEED)
        idx = rng.choice(N, size=n_sample, replace=False)
        pixels_fit = pixels[idx]
    else:
        pixels_fit = pixels

    # ── Fit GMM ──
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        max_iter=max_iter,
        random_state=GMM_RANDOM_SEED,
        warm_start=False,
    )
    gmm.fit(pixels_fit)

    # ── Predict trên toàn bộ pixel ──
    # predict_proba trả về posterior P(cluster_k | pixel)
    proba = gmm.predict_proba(pixels)          # (H*W, K)

    # ── Shannon Entropy: H = -sum(p * log(p)) ──
    eps     = 1e-8
    entropy = -np.sum(proba * np.log(proba + eps), axis=1)  # (H*W,)

    # Normalize về [0, 1] bằng entropy tối đa = log(K)
    entropy_max = np.log(n_components)
    uncertainty = (entropy / (entropy_max + eps)).reshape(H, W)
    uncertainty = np.clip(uncertainty, 0.0, 1.0).astype(np.float32)
    return uncertainty


# ── Worker function cho multiprocessing ──────────────────────────────────────
def process_one_image(args):
    """Chạy GMM trên 1 ảnh, return (unc_path, unc_map) hoặc raise."""
    img_path, unc_path = args
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Cannot read: {img_path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    unc = compute_gmm_uncertainty(img)
    return unc_path, unc, img.shape[:2]


# ── Canny Edge (GPU batch) ───────────────────────────────────────────────────
def compute_canny_batch(imgs_tensor):
    """
    Args:
        imgs_tensor: (B, 3, H, W) float32 GPU tensor [0, 1]
    Returns:
        (B, H, W) float32 GPU tensor {0, 1}
    """
    # Tính Canny trên từng kênh R, G, B rồi lấy hợp (max)
    # Đây là cách tốt nhất để không bỏ sót biên màu sắc
    cR = kornia.filters.canny(imgs_tensor[:, 0:1], low_threshold=CANNY_LOW, high_threshold=CANNY_HIGH)[1]
    cG = kornia.filters.canny(imgs_tensor[:, 1:2], low_threshold=CANNY_LOW, high_threshold=CANNY_HIGH)[1]
    cB = kornia.filters.canny(imgs_tensor[:, 2:3], low_threshold=CANNY_LOW, high_threshold=CANNY_HIGH)[1]
    edge_map = torch.max(torch.max(cR, cG), cB) # (B, 1, H, W)
    return edge_map.squeeze(1).float()          # (B, H, W)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f'Thiết bị GPU: {DEVICE}')
    print(f'GMM: {GMM_N_COMPONENTS} cụm, subsample={GMM_N_SAMPLE:,}, workers={NUM_WORKERS}')

    # Thu thập danh sách ảnh
    all_imgs = set()
    for split in SPLITS:
        split_file = osp.join(DATA_ROOT, split)
        if not osp.exists(split_file):
            print(f'WARNING: {split_file} không tồn tại, bỏ qua.')
            continue
        with open(split_file) as f:
            for line in f:
                name = line.strip()
                if name:
                    all_imgs.add(name)

    all_imgs = sorted(all_imgs)
    print(f'Tổng số ảnh trong split: {len(all_imgs)}')

    # ── Xóa cache cũ nếu FORCE_RECOMPUTE ──
    if FORCE_RECOMPUTE:
        import shutil
        print('\n🧹 Xóa uncertainty/ và edges/ cũ...')
        cities = set()
        for img_name in all_imgs:
            basename = osp.splitext(osp.basename(img_name))[0]
            city = get_city(DATA_ROOT, basename)
            if city:
                cities.add(city)
        for city in cities:
            for folder in ['uncertainty', 'edges']:
                p = osp.join(DATA_ROOT, city, folder)
                if osp.exists(p):
                    shutil.rmtree(p)
        print(f'✨ Đã dọn dẹp {len(cities)} city folders.\n')

    # ── Lọc ảnh chưa tính ──
    pending_unc   = []   # [(img_path, unc_path)]
    pending_edge  = []   # [(img_path, edge_path, H, W)]
    meta_all      = {}   # img_name → (city, basename, unc_path, edge_path)

    for img_name in all_imgs:
        basename = osp.splitext(osp.basename(img_name))[0]
        city = get_city(DATA_ROOT, basename)
        if city is None:
            print(f'SKIP (no city): {img_name}')
            continue

        img_path  = osp.join(DATA_ROOT, city, 'images',      basename + '.tif')
        unc_path  = osp.join(DATA_ROOT, city, 'uncertainty',  basename + '.npy')
        edge_path = osp.join(DATA_ROOT, city, 'edges',        basename + '.npy')

        os.makedirs(osp.dirname(unc_path),  exist_ok=True)
        os.makedirs(osp.dirname(edge_path), exist_ok=True)

        meta_all[img_name] = (city, basename, img_path, unc_path, edge_path)

        if not osp.exists(unc_path):
            pending_unc.append((img_path, unc_path))
        if not osp.exists(edge_path):
            pending_edge.append((img_path, edge_path))

    print(f'Cần tính uncertainty: {len(pending_unc)} ảnh')
    print(f'Cần tính edges:       {len(pending_edge)} ảnh\n')

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 1: GMM Uncertainty (CPU, có thể song song hóa)
    # ════════════════════════════════════════════════════════════════════════
    if pending_unc:
        print('═' * 60)
        print('PHASE 1: Tính GMM Uncertainty')
        print('═' * 60)

        errors = 0
        if NUM_WORKERS <= 1:
            # Sequential mode (dễ debug)
            for img_path, unc_path in tqdm(pending_unc, desc='GMM (seq)'):
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        errors += 1
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    unc = compute_gmm_uncertainty(img)
                    np.save(unc_path, unc)
                except Exception as e:
                    print(f'  ERROR {img_path}: {e}')
                    errors += 1
        else:
            # Parallel mode với ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = {
                    executor.submit(process_one_image, args): args
                    for args in pending_unc
                }
                with tqdm(total=len(pending_unc), desc='GMM (parallel)') as pbar:
                    for future in as_completed(futures):
                        try:
                            unc_path, unc, shape = future.result()
                            np.save(unc_path, unc)
                        except Exception as e:
                            print(f'  ERROR: {e}')
                            errors += 1
                        pbar.update(1)

        print(f'Phase 1 xong. Lỗi: {errors}\n')

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 2: Canny Edges (GPU batch)
    # ════════════════════════════════════════════════════════════════════════
    if pending_edge:
        print('═' * 60)
        print('PHASE 2: Tính Canny Edges (GPU batch)')
        print('═' * 60)

        errors = 0
        for batch_start in tqdm(range(0, len(pending_edge), CANNY_BATCH),
                                desc='Canny GPU'):
            batch = pending_edge[batch_start: batch_start + CANNY_BATCH]
            imgs_np, metas = [], []

            for img_path, edge_path in batch:
                img = cv2.imread(img_path)
                if img is None:
                    errors += 1
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                H, W = img.shape[:2]
                imgs_np.append(img)
                metas.append((edge_path, H, W))

            if not imgs_np:
                continue

            # Pad/resize để đồng nhất kích thước trong batch
            target_H, target_W = metas[0][1], metas[0][2]
            tensors = []
            for img in imgs_np:
                if img.shape[0] != target_H or img.shape[1] != target_W:
                    img = cv2.resize(img, (target_W, target_H))
                t = torch.from_numpy(img.astype(np.float32) / 255.0)
                tensors.append(t.permute(2, 0, 1))   # (3, H, W)

            batch_tensor = torch.stack(tensors).to(DEVICE)   # (B, 3, H, W)

            with torch.no_grad():
                edge_batch = compute_canny_batch(batch_tensor)   # (B, H, W)

            edge_cpu = edge_batch.cpu().numpy()

            for i, (edge_path, H, W) in enumerate(metas):
                if i >= len(edge_cpu):
                    break
                edge_map = edge_cpu[i].astype(np.float32)
                if edge_map.shape != (H, W):
                    edge_map = cv2.resize(edge_map, (W, H), interpolation=cv2.INTER_NEAREST)
                np.save(edge_path, edge_map)

        print(f'Phase 2 xong. Lỗi: {errors}\n')

    print('✅ Hoàn thành! Các .npy đã lưu vào:')
    print(f'   {{city}}/uncertainty/{{name}}.npy  ← GMM Entropy ∈ [0, 1]')
    print(f'   {{city}}/edges/{{name}}.npy        ← Canny binary {{0, 1}}')


if __name__ == '__main__':
    main()
