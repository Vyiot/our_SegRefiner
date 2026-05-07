import os
import warnings
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
warnings.filterwarnings('ignore')


"""
infer.py  —  GMM-guided refinement inference
==============================================
Pipeline:
  1. Chạy GMM (giống precompute_maps.py) trên ảnh RGB → unc_map ∈ [0,1]
  2. Tìm vùng có unc_map cao (pixel không tin cậy), sliding-window + NMS
     để chọn tối đa N patch 256×256
  3. Mỗi patch chạy full 6 bước diffusion denoising t=5→4→3→2→1→0 để sửa
  4. Weighted-blend kết quả patch vào mask tổng

Output: Bảng so sánh mIoU: Pseudo vs GMM-Refine
"""

import sys
import os.path as osp
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture

import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from mmcv.parallel import collate, scatter

ROOT_DIR         = osp.abspath(osp.join(osp.dirname(__file__), '..'))
CONFIG_FILE      = osp.join(ROOT_DIR, 'configs/segrefiner/exp8_all.py')
CHECKPOINT       = osp.join(ROOT_DIR, 'work_dirs/exp8_all_8/best_model1.pth')
VIS_DIR          = osp.join(ROOT_DIR, 'work_dirs/exp8_all_8/vis_gmm_refine')
DEVICE           = 'cuda:0'
BATCH_SIZE       = 1
NUM_WORKERS      = 4
SAVE_VIS         = True
VIS_MAX          = 9999   # lưu tất cả ảnh

# Override dataset (None = dùng val set trong config)
VAL_DATA_ROOT    = None  # None = dùng val set trong config (OEM_v2_Building/val.txt)
VAL_PSEUDO_DIR   = 'pseudolabels'   # tên thư mục pseudo-label trong VAL_DATA_ROOT

PATCH_SIZE       = 256          # kích thước patch local
UNC_THRESHOLD    = 0.3          # pixel có unc_map > ngưỡng này → cần sửa (giống RandomCropAll train)
MAX_LOCAL_PATCHES = 48          # số patch tối đa mỗi ảnh
NMS_IOU_THR      = 0.3          # NMS IoU threshold để lọc patch chồng lấp

# Chỉ chạy trên ảnh này để test nhanh (None = chạy tất cả)
TARGET_IMAGE     = None  # None = chạy tất cả



# ============================================================
# GMM Uncertainty (clone từ precompute_maps.py)
# ============================================================
def compute_gmm_uncertainty(img_rgb: np.ndarray) -> np.ndarray:
    """
    Eq. 1: M_unc = H(GMM(x_rgb)) / log(K)  ∈ [0, 1]
    - W = GMM với K tối ưu chọn bằng BIC (k=2..4)
    - H = entropy của posterior  = -Σ p·log(p)
    - chuẩn hoá bằng log(K)  (max entropy của K class)
    """
    H, W, C = img_rgb.shape
    pixels   = img_rgb.astype(np.float32).reshape(-1, C) / 255.0

    # Subsample để tăng tốc — dùng local RNG cố định để GMM deterministic
    rng      = np.random.RandomState(42)
    n_sub    = min(50_000, pixels.shape[0])
    idx      = rng.choice(pixels.shape[0], n_sub, replace=False)
    sub      = pixels[idx]

    # Chọn K tốt nhất bằng BIC
    best_k, best_bic, best_gmm = 2, np.inf, None
    for k in range(2, 5):
        gmm = GaussianMixture(n_components=k, covariance_type='full',
                              max_iter=50, random_state=42)
        gmm.fit(sub)
        bic = gmm.bic(sub)
        if bic < best_bic:
            best_bic, best_k, best_gmm = bic, k, gmm

    proba   = best_gmm.predict_proba(pixels)                   # (N, K)
    entropy = -np.sum(proba * np.log(proba + 1e-8), axis=1)   # (N,)
    unc     = (entropy / np.log(best_k)).reshape(H, W).astype(np.float32)
    return np.clip(unc, 0.0, 1.0)


# ============================================================
# Helpers
# ============================================================
def _iou_accum(pred_bin, gt_bin, ri, ru, ri_bg, ru_bg):
    p = pred_bin.astype(bool)
    g = gt_bin.astype(bool)
    ri    += np.count_nonzero(p & g)
    ru    += np.count_nonzero(p | g)
    ri_bg += np.count_nonzero(~p & ~g)
    ru_bg += np.count_nonzero(~p | ~g)
    return ri, ru, ri_bg, ru_bg


def _to_binary(arr):
    arr = np.squeeze(arr)
    if arr.dtype != np.uint8:
        arr = (arr >= 0.5).astype(np.uint8)
    else:
        arr = (arr > 0).astype(np.uint8)
    return arr


def compute_pseudo_iou(dataset, img_names_filter=None):
    pseudo_dir = osp.join(dataset.data_root, 'pseudolabels')
    label_dir  = osp.join(dataset.data_root, 'labels')
    ri, ru, ri_bg, ru_bg = 0, 0, 0, 0
    names = img_names_filter if img_names_filter is not None else dataset.img_names
    for img_name in names:
        basename = osp.splitext(img_name)[0]
        pseudo = cv2.imread(osp.join(pseudo_dir, img_name), cv2.IMREAD_GRAYSCALE)
        gt     = cv2.imread(osp.join(label_dir,  basename + '.tif'), cv2.IMREAD_GRAYSCALE)
        if pseudo is None or gt is None:
            continue
        if pseudo.shape != gt.shape:
            pseudo = cv2.resize(pseudo, (gt.shape[1], gt.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        c = (pseudo > 0)
        g = (gt == 1)
        ri    += np.count_nonzero(c & g)
        ru    += np.count_nonzero(c | g)
        ri_bg += np.count_nonzero(~c & ~g)
        ru_bg += np.count_nonzero(~c | ~g)
    return ri / max(ru, 1), ri_bg / max(ru_bg, 1)


def nms_patches(candidates, max_patches, nms_iou_thr):
    """Greedy NMS trên list (score, y1, x1, y2, x2)."""
    candidates = sorted(candidates, key=lambda c: -c[0])
    kept, suppressed = [], set()
    for i, (score, y1, x1, y2, x2) in enumerate(candidates):
        if len(kept) >= max_patches:
            break
        if i in suppressed:
            continue
        kept.append((y1, x1, y2, x2))
        for j in range(i + 1, len(candidates)):
            if j in suppressed:
                continue
            _, y1b, x1b, y2b, x2b = candidates[j]
            iy1, ix1 = max(y1, y1b), max(x1, x1b)
            iy2, ix2 = min(y2, y2b), min(x2, x2b)
            inter = max(0, iy2 - iy1) * max(0, ix2 - ix1)
            union = (y2-y1)*(x2-x1) + (y2b-y1b)*(x2b-x1b) - inter
            if inter / (union + 1e-6) > nms_iou_thr:
                suppressed.add(j)
    return kept


# ============================================================
# Core inference
# ============================================================
@torch.no_grad()
def infer_gmm_refine(model_obj, img_tensor, img_rgb_np, c_mask_np, device):
    """
    Pipeline:
      1. GMM trên img_rgb_np → unc_map (H×W float32 ∈ [0,1])
      2. Sliding-window + NMS trên unc_map → chọn patch cần sửa
      3. Mỗi patch: full 6-step denoising t=5→4→3→2→1→0
      4. Weighted-blend patch vào base_mask (= coarse mask)

    Args:
        model_obj   : SegRefiner model (unwrapped)
        img_tensor  : (1,3,H,W) normalized tensor trên device
        img_rgb_np  : (H,W,3) uint8 numpy  (denormalized, RGB)
        c_mask_np   : (H,W) uint8 binary coarse mask
        device      : torch device string

    Returns:
        result_np   : (H,W) uint8 binary refined mask
        unc_map     : (H,W) float32 uncertainty map (để visualize)
        vis_steps   : list of (label, (H,W) float32) — intermediate steps từ patch 0
    """
    H, W = img_tensor.shape[-2:]
    P    = PATCH_SIZE

    # ── Bước 1: GMM uncertainty từ RGB ──────────────────────────────────────
    unc_map = compute_gmm_uncertainty(img_rgb_np)   # (H,W) float32
    print(f'  [Step1-GMM] unc_map: min={unc_map.min():.3f} max={unc_map.max():.3f} '
          f'mean={unc_map.mean():.3f}  px>thr={((unc_map>UNC_THRESHOLD).mean()*100):.1f}%')

    # Chuyển unc_map sang tensor (1,1,H,W) — dùng làm fine_probs ban đầu cho Stage 2
    # Giống training: pixel uncertain cao → model được trust hơn để sửa
    unc_tensor = torch.from_numpy(unc_map).to(device).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    # ── Bước 2: Tìm patches centered on uncertain pixels ─────────────────────
    # Giống RandomCropAll trong train: pixel có unc > 0.3 làm tâm patch
    half_P = P // 2
    unc_mask_bin = (unc_map > UNC_THRESHOLD)
    ys_unc, xs_unc = np.where(unc_mask_bin)

    candidates = []
    if len(ys_unc) > 0:
        # Subsample tối đa 200 tâm để tránh tạo quá nhiều candidates
        rng_patch = np.random.RandomState(0)
        n_centers = min(200, len(ys_unc))
        chosen = rng_patch.choice(len(ys_unc), n_centers, replace=False)
        for i in chosen:
            cy, cx = int(ys_unc[i]), int(xs_unc[i])
            y1 = int(np.clip(cy - half_P, 0, max(0, H - P)))
            x1 = int(np.clip(cx - half_P, 0, max(0, W - P)))
            y2 = min(y1 + P, H)
            x2 = min(x1 + P, W)
            score = float(unc_map[y1:y2, x1:x2].mean())
            candidates.append((score, y1, x1, y2, x2))

    # Fallback: không có vùng uncertain → sliding window đơn giản
    if not candidates:
        stride = P // 2
        ys_sw = list(range(0, max(1, H - P + 1), stride)) or [0]
        xs_sw = list(range(0, max(1, W - P + 1), stride)) or [0]
        for y1 in ys_sw:
            for x1 in xs_sw:
                y2, x2 = min(y1 + P, H), min(x1 + P, W)
                score = float(unc_map[y1:y2, x1:x2].mean())
                candidates.append((score, y1, x1, y2, x2))

    kept = nms_patches(candidates, MAX_LOCAL_PATCHES, NMS_IOU_THR)
    print(f'  [Step2-NMS] candidates={len(candidates)}  kept_patches={len(kept)}')
    if kept:
        scores = sorted([c[0] for c in candidates], reverse=True)[:len(kept)]
        print(f'             patch scores (top): {[f"{s:.3f}" for s in scores[:5]]}')

    # ── Chuẩn bị tensor coarse mask (1,1,H,W) ───────────────────────────────
    c_tensor = torch.from_numpy(c_mask_np.astype(np.float32)).to(device)
    base_mask = c_tensor.unsqueeze(0).unsqueeze(0)   # (1,1,H,W)

    if not kept:
        print('  [SKIP] Không có patch cần sửa → trả thẳng coarse mask')
        return c_mask_np.copy(), unc_map, []

    # ── Bước 3 & 4: Với mỗi patch, chạy 6 bước t=5→0 rồi blend ─────────────
    # Cosine window để tránh vết cắt
    w_1d         = torch.sin(torch.linspace(0, np.pi, P, device=device))
    patch_weight = (w_1d.view(-1, 1) * w_1d.view(1, -1)).view(1, 1, P, P)

    accum_mask   = torch.zeros_like(base_mask, dtype=torch.float32)
    accum_weight = torch.zeros_like(base_mask, dtype=torch.float32)

    # Toàn bộ timestep: 5,4,3,2,1,0
    all_indices = list(range(model_obj.num_timesteps - 1, -1, -1))

    # Per-step accumulator: blend TẤT CẢ patches tại mỗi timestep để visualize
    _step_m = {i: torch.zeros_like(base_mask, dtype=torch.float32) for i in all_indices}
    _step_w = {i: torch.zeros_like(base_mask, dtype=torch.float32) for i in all_indices}

    for pidx, (y1, x1, y2, x2) in enumerate(kept):
        ph, pw = y2 - y1, x2 - x1
        coarse_mean_patch = float(base_mask[0, 0, y1:y2, x1:x2].mean().cpu())
        print(f'  [Patch {pidx}] bbox=({y1},{x1},{y2},{x2})  coarse_mean={coarse_mean_patch:.3f}')

        img_patch  = img_tensor[:, :, y1:y2, x1:x2]
        mask_patch = base_mask[:, :, y1:y2, x1:x2]
        fp_patch   = unc_tensor[:, :, y1:y2, x1:x2]   # fine_probs từ GMM

        # Pad nếu patch nhỏ hơn P (vùng biên ảnh)
        if ph < P or pw < P:
            img_patch  = F.pad(img_patch,  (0, P - pw, 0, P - ph))
            mask_patch = F.pad(mask_patch, (0, P - pw, 0, P - ph))
            fp_patch   = F.pad(fp_patch,   (0, P - pw, 0, P - ph))

        # ── Chạy full 6 bước denoising trên patch ────────────────────────
        cur_x          = mask_patch.clone()
        cur_fine_probs = fp_patch.clone()   # khởi tạo từ GMM unc_map, không phải zeros

        for i in all_indices:
            x_mean_before = float(cur_x.mean().cpu())
            t           = torch.tensor([i], device=device)
            model_input = torch.cat((img_patch, cur_x), dim=1)
            cur_x, cur_fine_probs = model_obj.p_sample(model_input, cur_fine_probs, t)

            if i == 0:
                cur_x = cur_x.sigmoid()
                print(f'    t={i}: logit_mean={x_mean_before:.4f} → sigmoid_mean={float(cur_x.mean()):.4f}'
                      f'  fine_probs_mean={float(cur_fine_probs.mean()):.4f}')
            else:
                # Bernoulli sampling theo công thức paper (Eq.11)
                fine_map     = (torch.rand_like(cur_fine_probs) < cur_fine_probs).float()
                pred_x_start = (cur_x >= 0).float()
                cur_x        = pred_x_start * fine_map + mask_patch * (1 - fine_map)
                print(f'    t={i}: logit_mean={x_mean_before:.4f} → x_mean={float(cur_x.mean()):.4f}'
                      f'  fine_map%={float(fine_map.mean())*100:.1f}%'
                      f'  pred_x_start%={float(pred_x_start.mean())*100:.1f}%')

            # Tích luũ vào per-step accum (tất cả patches)
            pw_ = patch_weight[:, :, :ph, :pw]
            _step_m[i][:, :, y1:y2, x1:x2] += cur_x[:, :, :ph, :pw].detach() * pw_
            _step_w[i][:, :, y1:y2, x1:x2] += pw_

        # cur_x là probability (sau sigmoid ở bước t=0)
        refined_prob = cur_x   # (1,1,P,P)
        refined_bin  = (refined_prob >= 0.5).float()
        coarse_bin   = (mask_patch[:, :, :ph, :pw] >= 0.5).float()
        diff_added   = float(((refined_bin[:,:,:ph,:pw] == 1) & (coarse_bin == 0)).float().mean()) * 100
        diff_removed = float(((refined_bin[:,:,:ph,:pw] == 0) & (coarse_bin == 1)).float().mean()) * 100
        print(f'    → refined_prob_mean={float(refined_prob.mean()):.4f}  '
              f'added={diff_added:.1f}%  removed={diff_removed:.1f}%')

        # Weighted blend vào accum
        p_weight = patch_weight[:, :, :ph, :pw]
        accum_mask[:, :, y1:y2, x1:x2]   += refined_prob[:, :, :ph, :pw] * p_weight
        accum_weight[:, :, y1:y2, x1:x2] += p_weight

    # Vùng không có patch → giữ nguyên coarse mask
    no_patch = (accum_weight == 0)
    accum_mask[no_patch]   = base_mask.float()[no_patch]
    accum_weight[no_patch] = 1.0

    result = (accum_mask / accum_weight)
    result_np = (result[0, 0] >= 0.5).cpu().numpy().astype(np.uint8)

    # Tổng kết diff so với coarse
    total_added   = float(((result_np == 1) & (c_mask_np == 0)).mean()) * 100
    total_removed = float(((result_np == 0) & (c_mask_np == 1)).mean()) * 100
    print(f'  [Blend-Final] total_added={total_added:.2f}%  total_removed={total_removed:.2f}%')

    # Build vis_steps
    vis_steps = []
    for i in all_indices:
        no_p = (_step_w[i] == 0)
        _step_m[i][no_p] = base_mask.float()[no_p]
        _step_w[i][no_p] = 1.0
        step_full = (_step_m[i] / _step_w[i])[0, 0].cpu().numpy()
        vis_steps.append((f't={i}', step_full))

    return result_np, unc_map, vis_steps


# ============================================================
# Main
# ============================================================
def main():
    cfg = Config.fromfile(CONFIG_FILE)

    # Override dataset nếu VAL_DATA_ROOT được chỉ định
    if VAL_DATA_ROOT:
        cfg.data.val.data_root = VAL_DATA_ROOT
        cfg.data.val.split_file = osp.join(VAL_DATA_ROOT, 'test1.txt')
        # Cập nhật pseudolabel_dir trong val_pipeline
        for step in cfg.data.val.pipeline:
            if step.get('type') == 'LoadOEMCoarseMasks':
                step['pseudolabel_dir'] = osp.join(VAL_DATA_ROOT, VAL_PSEUDO_DIR)
                break
        print(f'[Dataset Override] data_root = {VAL_DATA_ROOT}')
        print(f'[Dataset Override] pseudo    = {VAL_PSEUDO_DIR}/')

    # Build val dataset
    # Xóa vis cũ để tránh lẫn ảnh cũ
    if SAVE_VIS and osp.exists(VIS_DIR):
        import shutil
        shutil.rmtree(VIS_DIR)
        print(f'Cleared old vis: {VIS_DIR}')

    val_dataset = build_dataset(cfg.data.val)
    print(f'Val set: {len(val_dataset)} images')

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=lambda x: collate(x, samples_per_gpu=BATCH_SIZE),
        pin_memory=False,
    )

    # Build & load model
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, CHECKPOINT, map_location='cpu')
    model = model.to(DEVICE)
    model.eval()
    model_obj = model.module if hasattr(model, 'module') else model

    print(f'Loaded: {CHECKPOINT}')
    print(f'num_timesteps = {model_obj.num_timesteps}')
    print(f'betas_cumprod = {model_obj.betas_cumprod}')
    print(f'UNC_THRESHOLD = {UNC_THRESHOLD}  (GMM-based)')
    print(f'MAX_PATCHES   = {MAX_LOCAL_PATCHES}')
    print(f'ALL_STEPS     = t=5→4→3→2→1→0 (6 bước)\n')

    # Mean/std để denormalize ảnh cho GMM
    mean_np = np.array([123.675, 116.28,  103.53 ]).reshape(1, 1, 3)
    std_np  = np.array([58.395,  57.12,   57.375 ]).reshape(1, 1, 3)

    # Accumulators
    ri, ru, ri_bg, ru_bg = 0, 0, 0, 0
    total_num = 0
    vis_count = 0
    evaluated_names = []   # tên ảnh đã evaluate (để tính pseudo IoU đúng)
    label_dir = osp.join(val_dataset.data_root, 'labels')

    for batch_idx, data in enumerate(val_loader):
        # Lọc chỉ ảnh target
        try:
            _name = data['img_metas'].data[0][0].get('ori_filename', '')
        except Exception:
            _name = ''
        if TARGET_IMAGE and _name != TARGET_IMAGE:
            continue

        print(f'\n━━━ Image [{batch_idx+1}/{len(val_loader)}]  {_name} ━━━')
        gpu_id   = int(DEVICE.split(':')[-1]) if ':' in DEVICE else 0
        data_gpu = scatter(data, [gpu_id])[0]

        try:
            img_meta  = data['img_metas'].data[0][0]
            img_name  = img_meta.get('ori_filename', '')
            crop_bbox = img_meta.get('crop_bbox', None)
        except Exception:
            img_name  = ''
            crop_bbox = None

        # Unpack img & coarse mask
        img = data_gpu['img']
        if not torch.is_tensor(img): img = img[0]
        img = img.to(DEVICE)
        if img.dim() == 3: img = img.unsqueeze(0)

        coarse = data_gpu['coarse_masks']
        if isinstance(coarse, list) and len(coarse) > 0 and isinstance(coarse[0], list):
            coarse = coarse[0]
        c_mask_np = _to_binary(coarse[0].masks[0])

        # Denormalize ảnh để chạy GMM
        img_np = (img[0].cpu().numpy().transpose(1, 2, 0) * std_np + mean_np)
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)   # (H,W,3) uint8 RGB

        # ── GMM-guided refinement ────────────────────────────────────────────
        result_np, unc_map, vis_steps = infer_gmm_refine(
            model_obj, img, img_np, c_mask_np, DEVICE
        )

        # Load GT
        gt_mask = None
        if img_name:
            basename = osp.splitext(img_name)[0]
            gt_raw = cv2.imread(osp.join(label_dir, basename + '.tif'),
                                cv2.IMREAD_GRAYSCALE)
            if gt_raw is not None:
                gt_mask = (gt_raw == 1).astype(np.uint8)
        if gt_mask is None:
            gt_mask = _to_binary(c_mask_np)

        # Align pred về GT shape
        def align_pred(pred, gt):
            if crop_bbox is not None:
                h_full, w_full = gt.shape
                y1, x1, y2, x2 = crop_bbox
                pred_crop = cv2.resize(pred, (x2 - x1, y2 - y1),
                                       interpolation=cv2.INTER_NEAREST)
                pred_full = np.zeros((h_full, w_full), dtype=np.uint8)
                pred_full[y1:y2, x1:x2] = pred_crop
                return pred_full
            if pred.shape != gt.shape:
                return cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            return pred

        pred_aligned = align_pred(result_np, gt_mask)
        ri, ru, ri_bg, ru_bg = _iou_accum(pred_aligned, gt_mask, ri, ru, ri_bg, ru_bg)
        evaluated_names.append(img_name)
        total_num += 1

        # Per-image IoU log: Pseudo vs Refined + delta
        p = pred_aligned.astype(bool)
        g = gt_mask.astype(bool)
        _bld_ref = np.count_nonzero(p & g) / max(np.count_nonzero(p | g), 1) * 100

        # Resize coarse mask về GT shape nếu cần
        c_aligned = c_mask_np
        if c_aligned.shape != gt_mask.shape:
            c_aligned = cv2.resize(c_aligned, (gt_mask.shape[1], gt_mask.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        c = c_aligned.astype(bool)
        _bld_pseudo = np.count_nonzero(c & g) / max(np.count_nonzero(c | g), 1) * 100

        _delta = _bld_ref - _bld_pseudo
        _sign  = '+' if _delta >= 0 else ''
        print(f'  [IoU vs GT]  pseudo={_bld_pseudo:.2f}%  refined={_bld_ref:.2f}%  Δ={_sign}{_delta:.2f}%')

        # ── Visualization: 1 ảnh duy nhất chứa toàn bộ pipeline ────────────
        if SAVE_VIS:
            import torchvision.utils as vutils
            from PIL import Image, ImageDraw

            S = 256  # panel size

            def to_t(arr_hw):
                a = arr_hw.astype(np.float32)
                if a.max() > 1.0: a = (a > 0).astype(np.float32)
                t = torch.from_numpy(a).unsqueeze(0).unsqueeze(0)
                return F.interpolate(t, (S, S), mode='nearest')[0].repeat(3, 1, 1)

            def img_to_t(arr_hwc):
                t = torch.from_numpy(arr_hwc.astype(np.float32) / 255.0
                                     ).permute(2, 0, 1).unsqueeze(0)
                return F.interpolate(t, (S, S), mode='bilinear', align_corners=False)[0]

            def unc_to_t(unc_hw):
                u8 = (np.clip(unc_hw, 0, 1) * 255).astype(np.uint8)
                colored = cv2.applyColorMap(u8, cv2.COLORMAP_VIRIDIS)
                colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
                t = torch.from_numpy(colored.astype(np.float32) / 255.0
                                     ).permute(2, 0, 1).unsqueeze(0)
                return F.interpolate(t, (S, S), mode='bilinear', align_corners=False)[0]

            def diff_to_t(pseudo_hw, refined_hw):
                p = _to_binary(pseudo_hw).astype(bool)
                r = _to_binary(refined_hw).astype(bool)
                diff_rgb = np.zeros((*p.shape, 3), dtype=np.uint8)
                diff_rgb[ p &  r] = [255, 255, 255]
                diff_rgb[~p & ~r] = [  0,   0,   0]
                diff_rgb[ p & ~r] = [255,  50,  50]
                diff_rgb[~p &  r] = [ 50, 220,  50]
                t = torch.from_numpy(diff_rgb.astype(np.float32) / 255.0
                                     ).permute(2, 0, 1).unsqueeze(0)
                return F.interpolate(t, (S, S), mode='nearest')[0]

            # Panels: RGB | Pseudo | GMM-Unc | t=5..t=0 | Refined | Diff | GT
            panels = [img_to_t(img_np), to_t(c_mask_np), unc_to_t(unc_map)]
            labels = ['RGB', 'Pseudo', 'GMM-Unc']

            for lbl, step_arr in vis_steps:
                panels.append(to_t(step_arr))
                labels.append(lbl + '(all)')

            panels.append(to_t(result_np))
            labels.append('Refined')

            panels.append(diff_to_t(c_mask_np, result_np))
            labels.append('Diff(R=del G=add)')

            panels += [to_t(gt_mask)]
            labels += ['GT']

            grid  = vutils.make_grid(panels, nrow=len(panels), padding=4, pad_value=0.5)
            ndarr = grid.mul(255).clamp(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
            im    = Image.fromarray(ndarr)
            draw  = ImageDraw.Draw(im)
            for k, lbl in enumerate(labels):
                draw.text((k * (S + 4) + 4, 4), lbl, fill=(255, 220, 0))

            os.makedirs(VIS_DIR, exist_ok=True)
            name = osp.splitext(img_name)[0] if img_name else f'img{batch_idx:04d}'
            im.save(osp.join(VIS_DIR, f'{name}.png'))
            vis_count += 1

        # Progress
        if (batch_idx + 1) % 50 == 0:
            b_run  = ri / max(ru,  1) * 100
            bg_run = ri_bg / max(ru_bg, 1) * 100
            print(f'  [{batch_idx+1}/{len(val_loader)}]  '
                  f'bld={b_run:.2f}%  bg={bg_run:.2f}%  '
                  f'mIoU={(b_run+bg_run)/2:.2f}%')

    # ── Final metrics ─────────────────────────────────────────────────────────
    iou_bld = ri    / max(ru,    1)
    iou_bg  = ri_bg / max(ru_bg, 1)
    miou    = (iou_bld + iou_bg) / 2

    pseudo_b, pseudo_bg = compute_pseudo_iou(val_dataset, img_names_filter=evaluated_names)
    pseudo_m = (pseudo_b + pseudo_bg) / 2

    try:
        from terminaltables import AsciiTable
        table_data = [
            ['Class',      'GMM-Refined',          'Pseudo (input)'],
            ['background', f'{iou_bg*100:.2f}',     f'{pseudo_bg*100:.2f}'],
            ['building',   f'{iou_bld*100:.2f}',    f'{pseudo_b*100:.2f}'],
            ['mIoU',       f'{miou*100:.2f}',        f'{pseudo_m*100:.2f}'],
            ['Δ vs Pseudo', f'{(miou-pseudo_m)*100:+.2f}', '-'],
        ]
        print('\n' + AsciiTable(table_data).table)
    except ImportError:
        print(f'\n{"="*50}')
        print(f'{"Mode":<18} {"bg IoU":>8} {"bld IoU":>9} {"mIoU":>7}')
        print(f'{"-"*50}')
        print(f'{"GMM-Refined":<18} {iou_bg*100:>8.2f} {iou_bld*100:>9.2f} {miou*100:>7.2f}')
        print(f'{"Pseudo (input)":<18} {pseudo_bg*100:>8.2f} {pseudo_b*100:>9.2f} {pseudo_m*100:>7.2f}')
        print(f'{"-"*50}')
        print(f'Δ GMM vs Pseudo: {(miou-pseudo_m)*100:+.2f}%')
        print(f'{"="*50}')

    print(f'\nImages evaluated: {total_num}')
    if SAVE_VIS:
        print(f'Visualizations saved to: {VIS_DIR}')


if __name__ == '__main__':
    main()
