"""
Precompute uncertainty maps + Canny edge maps (LEVEL 4 CLEANUP)
- Option to skip GMM and only recompute Canny
"""
import os
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from concurrent.futures import ProcessPoolExecutor, as_completed

# ─── CẤU HÌNH ĐIỀU KHIỂN ─────────────────────────────────────────────────────
DATA_ROOT       = '/home/ubuntu/vy/Denoiser/OpenEarthMap_wo_xBD'
SPLITS          = ['train.txt']

# CẤU HÌNH CHẠY LẠI
RECOMPUTE_UNCERTAINTY = False  # ĐỂ FALSE NẾU KHÔNG MUỐN CHẠY LẠI GMM (TIẾT KIỆM 3-5 TIẾNG)
RECOMPUTE_CANNY       = True   # ĐỂ TRUE ĐỂ LÀM SẠCH BIÊN (LEVEL 4)

# THÔNG SỐ CANNY SIÊU SẠCH
CANNY_LOW       = 120
CANNY_HIGH      = 250
MIN_EDGE_AREA   = 500

NUM_WORKERS      = 8           # Đa nhân CPU
# ──────────────────────────────────────────────────────────────────────────────

def get_city(data_root, basename):
    parts = basename.split('_')
    for i in range(len(parts), 0, -1):
        candidate = '_'.join(parts[:i])
        if osp.isdir(osp.join(data_root, candidate)):
            return candidate
    return None

def compute_gmm_uncertainty(img_rgb):
    H, W, C = img_rgb.shape
    img_norm = img_rgb.astype(np.float32) / 255.0
    pixels   = img_norm.reshape(-1, C)
    gmm = GaussianMixture(n_components=2, covariance_type='full', max_iter=100, random_state=42)
    # Subsample to speed up
    idx = np.random.choice(pixels.shape[0], min(50_000, pixels.shape[0]), replace=False)
    gmm.fit(pixels[idx])
    proba = gmm.predict_proba(pixels)
    entropy = -np.sum(proba * np.log(proba + 1e-8), axis=1)
    return (entropy / np.log(2)).reshape(H, W).astype(np.float32)

def compute_canny_clean(img_rgb):
    img_blur = cv2.GaussianBlur(img_rgb, (5, 5), 0)
    cR = cv2.Canny(img_blur[:,:,0], CANNY_LOW, CANNY_HIGH)
    cG = cv2.Canny(img_blur[:,:,1], CANNY_LOW, CANNY_HIGH)
    cB = cv2.Canny(img_blur[:,:,2], CANNY_LOW, CANNY_HIGH)
    edge_raw = cv2.bitwise_or(cv2.bitwise_or(cR, cG), cB)
    kernel = np.ones((3, 3), np.uint8)
    edge_conn = cv2.dilate(edge_raw, kernel, iterations=1)
    nb, out, stats, _ = cv2.connectedComponentsWithStats(edge_conn, connectivity=8)
    res = np.zeros(edge_raw.shape, dtype=np.uint8)
    for j in range(1, nb):
        if stats[j, cv2.CC_STAT_AREA] >= MIN_EDGE_AREA:
            res[out == j] = 255
    return (res.astype(np.float32) / 255.0)

def process_one_image(args):
    img_path, unc_path, edge_path = args
    img_bgr = cv2.imread(img_path)
    if img_bgr is None: return False
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 1. Chỉ tính GMM nếu được yêu cầu hoặc file chưa tồn tại
    if RECOMPUTE_UNCERTAINTY or not osp.exists(unc_path):
        unc = compute_gmm_uncertainty(img_rgb)
        np.save(unc_path, unc)
    
    # 2. Chỉ tính Canny nếu được yêu cầu hoặc file chưa tồn tại
    if RECOMPUTE_CANNY or not osp.exists(edge_path):
        edge = compute_canny_clean(img_rgb)
        np.save(edge_path, edge)
    
    return True

def main():
    all_imgs = set()
    for split in SPLITS:
        split_file = osp.join(DATA_ROOT, split)
        if not osp.exists(split_file): continue
        with open(split_file) as f:
            for line in f:
                name = line.strip()
                if name: all_imgs.add(name)
    
    all_imgs = sorted(all_imgs)
    pending = []
    for img_name in all_imgs:
        basename = osp.splitext(osp.basename(img_name))[0]
        city = get_city(DATA_ROOT, basename)
        if not city: continue
        img_path = osp.join(DATA_ROOT, city, 'images', basename + '.tif')
        unc_path = osp.join(DATA_ROOT, city, 'uncertainty', basename + '.npy')
        edge_path = osp.join(DATA_ROOT, city, 'edges', basename + '.npy')
        os.makedirs(osp.dirname(unc_path), exist_ok=True)
        os.makedirs(osp.dirname(edge_path), exist_ok=True)
        
        need_unc = RECOMPUTE_UNCERTAINTY or not osp.exists(unc_path)
        need_edge = RECOMPUTE_CANNY or not osp.exists(edge_path)
        
        if need_unc or need_edge:
            pending.append((img_path, unc_path, edge_path))

    if pending:
        print(f"Processing {len(pending)} images (GMM={RECOMPUTE_UNCERTAINTY}, Canny={RECOMPUTE_CANNY})")
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            list(tqdm(executor.map(process_one_image, pending), total=len(pending)))

    print("✅ Completed!")

if __name__ == '__main__':
    main()
