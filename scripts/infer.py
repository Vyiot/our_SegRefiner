import os
import warnings
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'   # tắt TIFF WARN của OpenCV
warnings.filterwarnings('ignore')          # tắt Python warnings (mmcv v2 notice...)

"""
infer_global_only.py
====================
Inference thử nghiệm: Chạy toàn bộ T timesteps (t=5→0) ở global scale (256x256),
KHÔNG có giai đoạn 2 (Local Patch Refinement).

Mục đích: So sánh với inference 2 giai đoạn tiêu chuẩn để đánh giá
xem giai đoạn local có thực sự cải thiện hay không.

Config gốc:  configs/segrefiner/exp8_all.py
Checkpoint:  work_dirs/exp8_all/best_model.pth
Data:        OEM_v2_Building (val set) – pipeline giống lúc train

Chạy từ thư mục SegRefiner/:
    python scripts/infer_global_only.py

Output:
    - In bảng mIoU so với Pseudo IoU
    - Lưu ảnh visualization vào work_dirs/exp8_all/vis_global_only/
"""

import sys
import os.path as osp

# Đảm bảo import đúng mmdet custom
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from mmcv.parallel import collate, scatter

# ============================================================
# Cấu hình
# ============================================================
CONFIG_FILE  = 'configs/segrefiner/exp8_all.py'
CHECKPOINT   = 'work_dirs/exp8_all/latest.pth'
VIS_DIR      = 'work_dirs/exp8_all/vis_latest'
DEVICE       = 'cuda:0'
BATCH_SIZE   = 1          # Val luôn dùng batch=1
NUM_WORKERS  = 4
SAVE_VIS     = True       # Lưu ảnh visualization cho N ảnh đầu
VIS_MAX      = 10         # Số ảnh tối đa lưu visualization

# Dataset override (để trống '' → dùng từ config gốc)
DATA_ROOT  = ''
SPLIT_FILE = ''
PSEUDO_DIR = ''

# ============================================================
# Helper: tính IoU
# ============================================================
def _iou_accum(pred_bin, gt_bin, ri, ru, ri_bg, ru_bg):
    """Cộng dồn (intersection, union) cho cả building và background."""
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


# ============================================================
# Inference: 2-stage
#   Stage 1 (Global, 256×256): chạy full t=5→0 → xác định fine_probs
#   Stage 2 (Local, 1024×1024): crop patch uncertain → chạy lại full t=5→0
# ============================================================
@torch.no_grad()
def infer_global_only(model, data, device):
    model_obj = model.module if hasattr(model, 'module') else model

    # ── Unpack data ─────────────────────────────────────────
    def _undc(v):
        if hasattr(v, 'data'): return v.data[0]
        return v

    img = _undc(data['img'])
    if not torch.is_tensor(img): img = img[0]
    img = img.to(device)
    if img.dim() == 3: img = img.unsqueeze(0)    # (1,3,1024,1024)

    coarse = _undc(data['coarse_masks'])
    if isinstance(coarse, list) and len(coarse) > 0 and isinstance(coarse[0], list):
        coarse = coarse[0]

    c_mask_np = coarse[0].masks[0]    # (H,W) numpy
    if c_mask_np.sum() <= 128:
        return _to_binary(c_mask_np), _to_binary(c_mask_np)

    img_h, img_w = img.shape[-2:]   # 1024, 1024
    T       = model_obj.num_timesteps          # 6
    indices = list(range(T))[::-1]            # [5,4,3,2,1,0]

    c_tensor = torch.from_numpy(c_mask_np).to(device).float()  # (H,W)
    c_4d     = c_tensor.unsqueeze(0).unsqueeze(0)              # (1,1,H,W)

    # ══════════════════════════════════════════════════════
    # STAGE 1: Global 256×256 — chỉ 1 bước t=5 → lấy fine_probs
    # ══════════════════════════════════════════════════════
    img_256  = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
    mask_256 = F.interpolate(c_4d, size=(256, 256), mode='nearest')

    cur_x  = mask_256.clone()
    cur_fp = torch.zeros_like(mask_256)   # (1,1,256,256)

    # Chỉ 1 bước tại t=T-1 (t=5)
    t_idx    = indices[0]   # = 5
    t_in     = torch.tensor([t_idx], device=device)
    model_in = torch.cat((img_256, cur_x), dim=1)
    pred_lg  = model_obj.denoise_model(model_in, t_in)

    x_fp    = 2 * torch.abs(pred_lg.sigmoid() - 0.5)
    beta_t  = model_obj.betas_cumprod[t_idx]
    beta_tp = model_obj.betas_cumprod_prev[t_idx]
    p_cf    = x_fp * (beta_tp - beta_t) / (1 - x_fp * beta_t + 1e-6)
    cur_fp  = cur_fp + (1 - cur_fp) * p_cf
    # cur_x không cần cập nhật — chỉ dùng cur_fp để tìm patch

    # Upscale global result & fine_probs lên 1024
    global_prob_1024 = F.interpolate(cur_x,  size=(img_h, img_w), mode='bilinear', align_corners=False)
    fine_probs_1024  = F.interpolate(cur_fp, size=(img_h, img_w), mode='bilinear', align_corners=False)
    global_bin_1024  = (global_prob_1024 >= 0.5).float()   # (1,1,1024,1024)

    # ══════════════════════════════════════════════════════
    # STAGE 2: Local 1024×1024 — tìm patch uncertain → full t=5→0
    # ══════════════════════════════════════════════════════
    fine_prob_thr = model_obj.test_cfg.get('fine_prob_thr', 0.95)
    iou_thr       = model_obj.test_cfg.get('iou_thr', 0.15)
    batch_max     = model_obj.test_cfg.get('batch_max', 32)
    PATCH         = model_obj.test_cfg.get('model_size', 256)

    # Tìm vùng ít tự tin (từ fine_probs 256×256 của stage 1)
    thr_val  = cur_fp.max().item() * fine_prob_thr
    low_conf = cur_fp < thr_val                               # (1,1,256,256)
    y_c, x_c = torch.where(low_conf.squeeze(0).squeeze(0))   # coords trong 256

    if y_c.numel() == 0:
        return (global_bin_1024[0,0].cpu().numpy().astype(np.uint8), c_mask_np)

    # Scale coords 256 → 1024
    sy = img_h / cur_fp.shape[-2]
    sx = img_w / cur_fp.shape[-1]
    y_c = (y_c * sy).long()
    x_c = (x_c * sx).long()
    scores_cpu = (1 - fine_probs_1024.squeeze()[y_c, x_c]).cpu().float()

    # Tạo patch bbox 256×256 tâm vào pixel uncertain
    y1 = (y_c - PATCH // 2).float().clamp(0, img_h - PATCH);  y2 = y1 + PATCH
    x1 = (x_c - PATCH // 2).float().clamp(0, img_w - PATCH);  x2 = x1 + PATCH
    proposals = torch.stack((x1, y1, x2, y2), dim=-1).cpu().float()
    if scores_cpu.dim() == 0: scores_cpu = scores_cpu.unsqueeze(0)

    from mmcv.ops import nms as mmcv_nms
    patch_coors, _ = mmcv_nms(proposals, scores_cpu, iou_threshold=iou_thr)
    patch_coors = patch_coors.to(device).int()

    # Crop patches từ PSEUDO LABEL GỐC (c_4d), không phải stage 1 output
    valid_imgs, valid_masks, valid_coors = [], [], []
    for coor in patch_coors:
        x1c, y1c, x2c, y2c = coor[0], coor[1], coor[2], coor[3]
        pm = c_4d[:, :, y1c:y2c, x1c:x2c]   # (1,1,256,256) — pseudo label gốc
        if pm.any() and not pm.all():
            valid_imgs.append(img[:, :, y1c:y2c, x1c:x2c])
            valid_masks.append(pm)
            valid_coors.append(coor)

    if len(valid_imgs) == 0:
        return (global_bin_1024[0,0].cpu().numpy().astype(np.uint8), c_mask_np)

    patch_imgs_t  = torch.cat(valid_imgs,  dim=0)   # (N,3,256,256)
    patch_masks_t = torch.cat(valid_masks, dim=0)   # (N,1,256,256)

    # Chạy full t=5→0 trên từng batch patch
    local_results = []
    for start in range(0, len(valid_coors), batch_max):
        end    = min(len(valid_coors), start + batch_max)
        b_img  = patch_imgs_t[start:end]
        b_mask = patch_masks_t[start:end]

        cur_xb  = b_mask.clone()
        cur_fpb = torch.zeros_like(cur_xb)

        for step_i, t_idx in enumerate(indices):     # t=5→0
            t_b     = torch.tensor([t_idx] * b_img.shape[0], device=device)
            m_in    = torch.cat((b_img, cur_xb), dim=1)
            pred_lg = model_obj.denoise_model(m_in, t_b)

            x_fp    = 2 * torch.abs(pred_lg.sigmoid() - 0.5)
            beta_t  = model_obj.betas_cumprod[t_idx]
            beta_tp = model_obj.betas_cumprod_prev[t_idx]
            p_cf    = x_fp * (beta_tp - beta_t) / (1 - x_fp * beta_t + 1e-6)
            cur_fpb = cur_fpb + (1 - cur_fpb) * p_cf

            is_last = (step_i == len(indices) - 1)
            if is_last:
                cur_xb = pred_lg.sigmoid()
            else:
                noise    = torch.rand_like(pred_lg)
                fine_map = (noise < cur_fpb).float()
                pred_bin = (pred_lg >= 0).float()
                cur_xb   = pred_bin * fine_map + b_mask * (1 - fine_map)

        local_results.append(cur_xb)   # (B,1,256,256)

    local_all = torch.cat(local_results, dim=0)   # (N,1,256,256)

    # Paste patches vào pseudo label gốc (c_4d làm base)
    refined = c_4d[0, 0].clone()    # (1024,1024) — bắt đầu từ pseudo label
    weight  = torch.zeros_like(refined)
    for lm, coor in zip(local_all, valid_coors):
        x1c, y1c, x2c, y2c = coor[0].item(), coor[1].item(), coor[2].item(), coor[3].item()
        refined[y1c:y2c, x1c:x2c] += lm[0]
        weight[y1c:y2c, x1c:x2c]  += 1

    refined_area = (weight > 0).float()
    weight[weight == 0] = 1
    refined = (refined / weight >= 0.5).float()
    # Vùng ko có patch: giữ nguyên pseudo label gốc
    final   = refined_area * refined + (1 - refined_area) * c_4d[0, 0]

    return (final.cpu().numpy().astype(np.uint8), c_mask_np)


# ============================================================
# Pseudo IoU từ disk
# ============================================================
def compute_pseudo_iou(dataset):
    pseudo_dir = PSEUDO_DIR if PSEUDO_DIR else osp.join(dataset.data_root, 'pseudolabels')
    label_dir  = osp.join(dataset.data_root, 'labels')
    ri, ru, ri_bg, ru_bg = 0, 0, 0, 0
    for img_name in dataset.img_names:
        basename = osp.splitext(img_name)[0]
        pseudo = cv2.imread(osp.join(pseudo_dir, img_name), cv2.IMREAD_GRAYSCALE)
        gt     = cv2.imread(osp.join(label_dir, basename + '.tif'), cv2.IMREAD_GRAYSCALE)
        if pseudo is None or gt is None:
            continue
        if pseudo.shape != gt.shape:
            pseudo = cv2.resize(pseudo, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        c = (pseudo > 0)
        g = (gt == 1)
        ri    += np.count_nonzero(c & g)
        ru    += np.count_nonzero(c | g)
        ri_bg += np.count_nonzero(~c & ~g)
        ru_bg += np.count_nonzero(~c | ~g)
    iou_b  = ri    / max(ru, 1)
    iou_bg = ri_bg / max(ru_bg, 1)
    return iou_b, iou_bg


# ============================================================
# Collect intermediate steps tại 256×256 để visualize
# ============================================================
@torch.no_grad()
def get_step_intermediates(model, img_tensor, mask_np, device):
    """
    Chạy full t=5→0 ở 256×256 và trả về list 6 mask sau từng bước.
    img_tensor: (1,3,H,W) tensor chuẩn hóa
    mask_np:    (H,W) numpy uint8
    """
    model_obj = model.module if hasattr(model, 'module') else model
    T       = model_obj.num_timesteps
    indices = list(range(T))[::-1]   # [5,4,3,2,1,0]

    img_256  = F.interpolate(img_tensor, size=(256, 256), mode='bilinear', align_corners=False)
    c_tensor = torch.from_numpy(mask_np.astype(np.float32)).to(device)
    mask_256 = F.interpolate(c_tensor.unsqueeze(0).unsqueeze(0), size=(256, 256), mode='nearest')

    cur_x  = mask_256.clone()
    cur_fp = torch.zeros_like(mask_256)
    steps  = []

    for step_i, t_idx in enumerate(indices):
        t_in     = torch.tensor([t_idx], device=device)
        model_in = torch.cat((img_256, cur_x), dim=1)
        pred_lg  = model_obj.denoise_model(model_in, t_in)

        x_fp    = 2 * torch.abs(pred_lg.sigmoid() - 0.5)
        beta_t  = model_obj.betas_cumprod[t_idx]
        beta_tp = model_obj.betas_cumprod_prev[t_idx]
        p_cf    = x_fp * (beta_tp - beta_t) / (1 - x_fp * beta_t + 1e-6)
        cur_fp  = cur_fp + (1 - cur_fp) * p_cf

        is_last = (step_i == len(indices) - 1)
        if is_last:
            cur_x = pred_lg.sigmoid()
        else:
            noise    = torch.rand_like(pred_lg)
            fine_map = (noise < cur_fp).float()
            pred_bin = (pred_lg >= 0).float()
            cur_x    = pred_bin * fine_map + mask_256 * (1 - fine_map)

        # Lưu state sau bước này (binary)
        step_mask = (cur_x[0, 0].cpu().numpy() >= 0.5).astype(np.float32)
        steps.append(step_mask)   # 256×256

    return steps   # list 6 phần tử, mỗi phần tử là (256,256) float


# ============================================================
# Visualization helper
# ============================================================
def save_vis(img_rgb_np, pred_np, coarse_np, gt_np, save_path, steps=None):
    """
    Lưu strip:
      - steps=None: RGB | Pseudo | Pred | GT (4 panels)
      - steps=[...]: RGB | Pseudo | t5 | t4 | t3 | t2 | t1 | t0 | GT (9 panels)
    """
    import torchvision.utils as vutils
    from PIL import Image, ImageDraw

    def to_rgb_tensor(arr_hw):
        t = torch.from_numpy(arr_hw.astype(np.float32)).unsqueeze(0).repeat(3, 1, 1)
        return t

    def resize_t(t):
        return F.interpolate(t.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)[0]

    img_t    = torch.from_numpy(img_rgb_np.astype(np.float32) / 255.0).permute(2, 0, 1)
    coarse_t = to_rgb_tensor(coarse_np.astype(np.float32))
    pred_t   = to_rgb_tensor(pred_np.astype(np.float32))
    gt_t     = to_rgb_tensor(gt_np.astype(np.float32)) if gt_np is not None else pred_t

    if steps is not None and len(steps) > 0:
        # 9 panels: RGB | Pseudo | t5 | t4 | t3 | t2 | t1 | t0 | GT
        step_labels = [f't={t}' for t in reversed(range(len(steps)))]
        strip_list  = [resize_t(img_t), resize_t(coarse_t)]
        for s in steps:
            strip_list.append(resize_t(to_rgb_tensor(s)))
        strip_list.append(resize_t(gt_t))
        labels = ['RGB', 'Pseudo'] + step_labels + ['GT']
    else:
        strip_list = [resize_t(img_t), resize_t(coarse_t), resize_t(pred_t), resize_t(gt_t)]
        labels = ['RGB', 'Pseudo', 'Pred', 'GT']

    nrow = len(strip_list)
    grid = vutils.make_grid(strip_list, nrow=nrow, padding=4, pad_value=1.0)
    ndarr = grid.mul(255).clamp(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
    im    = Image.fromarray(ndarr)
    draw  = ImageDraw.Draw(im)
    for idx, lbl in enumerate(labels):
        draw.text((idx * (256 + 4) + 10, 10), lbl, fill=(255, 0, 0))
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    im.save(save_path)
    print(f'  >> Saved: {save_path}')


# ============================================================
# Main
# ============================================================
def main():
    # ── Load config ──────────────────────────────────────────
    cfg = Config.fromfile(CONFIG_FILE)

    # ── Build val dataset ────────────────────────────────────
    val_cfg = cfg.data.val

    # Override dataset nếu DATA_ROOT được set
    if DATA_ROOT:
        val_cfg = val_cfg.copy()
        val_cfg.data_root  = DATA_ROOT
        val_cfg.split_file = SPLIT_FILE
        # Cập nhật pseudolabel_dir trong pipeline
        for step in val_cfg.pipeline:
            if step.get('type') == 'LoadOEMCoarseMasks':
                step['pseudolabel_dir'] = PSEUDO_DIR
                break

    val_dataset = build_dataset(val_cfg)
    print(f'Val set (test_oem_raw): {len(val_dataset)} images')

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=lambda x: collate(x, samples_per_gpu=BATCH_SIZE),
        pin_memory=False,
    )

    # ── Build model ──────────────────────────────────────────
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, CHECKPOINT, map_location='cpu')
    model = model.to(DEVICE)
    model.eval()
    print(f'Loaded checkpoint: {CHECKPOINT}')

    T = model.num_timesteps
    print(f'num_timesteps = {T} | betas_cumprod = {model.betas_cumprod}')

    # ── Accumulate metrics ───────────────────────────────────
    ri, ru, ri_bg, ru_bg = 0, 0, 0, 0
    total_num = 0
    vis_count = 0

    label_dir = osp.join(val_dataset.data_root, 'labels')

    for batch_idx, data in enumerate(val_loader):
        # scatter data lên GPU
        # scatter cần integer device index, không phải string 'cuda:0'
        gpu_id   = int(DEVICE.split(':')[-1]) if ':' in DEVICE else 0
        data_gpu = scatter(data, [gpu_id])[0] if isinstance(data, dict) else data

        try:
            img_meta  = data['img_metas'].data[0][0]
            img_name  = img_meta.get('ori_filename', '')
            crop_bbox = img_meta.get('crop_bbox', None)
        except Exception:
            img_name  = ''
            crop_bbox = None

        # ── Inference global-only ────────────────────────────
        pred_small, coarse_small = infer_global_only(model, data_gpu, DEVICE)

        # ── Load GT từ disk ──────────────────────────────────
        gt_mask = None
        if img_name:
            basename = osp.splitext(img_name)[0]
            gt_raw = cv2.imread(osp.join(label_dir, basename + '.tif'), cv2.IMREAD_GRAYSCALE)
            if gt_raw is not None:
                gt_mask = (gt_raw == 1).astype(np.uint8)

        # ── Xử lý crop_bbox (giống eval hook) ────────────────
        if gt_mask is not None and crop_bbox is not None:
            h_full, w_full = gt_mask.shape
            y1, x1, y2, x2 = crop_bbox
            crop_h, crop_w  = y2 - y1, x2 - x1
            pred_crop = cv2.resize(pred_small, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
            pred_full = np.zeros((h_full, w_full), dtype=np.uint8)
            pred_full[y1:y2, x1:x2] = pred_crop
            pred_mask = pred_full
        elif gt_mask is not None:
            if pred_small.shape != gt_mask.shape:
                pred_mask = cv2.resize(pred_small, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                pred_mask = pred_small
        else:
            pred_mask = pred_small
            gt_mask   = _to_binary(coarse_small)

        # ── Accumulate IoU ────────────────────────────────────
        ri, ru, ri_bg, ru_bg = _iou_accum(pred_mask, gt_mask, ri, ru, ri_bg, ru_bg)
        total_num += 1

        # ── Visualization ─────────────────────────────────────
        if SAVE_VIS and vis_count < VIS_MAX:
            # data_gpu['img'] đã là tensor (1,3,H,W) sau scatter
            img_raw_t = data_gpu['img']
            if not torch.is_tensor(img_raw_t):
                img_raw_t = img_raw_t[0]
            img_raw = img_raw_t[0].cpu()   # (3,H,W)
            mean = np.array([123.675, 116.28, 103.53]).reshape(3, 1, 1)
            std  = np.array([ 58.395,  57.12,  57.375]).reshape(3, 1, 1)
            img_np = (img_raw.numpy() * std + mean).clip(0, 255).astype(np.uint8)
            img_np = img_np.transpose(1, 2, 0)   # (H,W,3)
            name   = osp.splitext(img_name)[0] if img_name else f'img{batch_idx:04d}'

            # Collect 6 intermediate steps tại 256×256
            steps = get_step_intermediates(
                model, img_raw_t.to(DEVICE), _to_binary(coarse_small), DEVICE)

            save_vis(img_np, pred_mask, _to_binary(coarse_small), gt_mask,
                     osp.join(VIS_DIR, f'{name}_vis.png'), steps=steps)
            vis_count += 1

        if (batch_idx + 1) % 50 == 0:
            cur_iou_b  = ri  / max(ru,    1)
            cur_iou_bg = ri_bg / max(ru_bg, 1)
            cur_miou   = (cur_iou_b + cur_iou_bg) / 2.0
            print(f'  [{batch_idx+1}/{len(val_loader)}] running mIoU: {cur_miou*100:.2f}%')

    # ── Tính metrics cuối ─────────────────────────────────────
    pred_iou_b  = ri     / max(ru,    1)
    pred_iou_bg = ri_bg  / max(ru_bg, 1)
    pred_miou   = (pred_iou_b + pred_iou_bg) / 2.0

    pseudo_iou_b, pseudo_iou_bg = compute_pseudo_iou(val_dataset)
    pseudo_miou = (pseudo_iou_b + pseudo_iou_bg) / 2.0

    # ── Hiển thị bảng kết quả ────────────────────────────────
    try:
        from terminaltables import AsciiTable
        table_data = [
            ['Class',        'Global-Only IoU',            'Pseudo IoU'],
            ['background',   f'{pred_iou_bg*100:.2f}',     f'{pseudo_iou_bg*100:.2f}'],
            ['building',     f'{pred_iou_b*100:.2f}',      f'{pseudo_iou_b*100:.2f}'],
            ['Summary',      f'mIoU: {pred_miou*100:.2f}', f'mIoU: {pseudo_miou*100:.2f}'],
        ]
        print('\n' + AsciiTable(table_data).table)
    except ImportError:
        print(f'\n=== Kết quả ===')
        print(f'  Global-Only  | bg: {pred_iou_bg*100:.2f}%  | building: {pred_iou_b*100:.2f}%  | mIoU: {pred_miou*100:.2f}%')
        print(f'  Pseudo (base)| bg: {pseudo_iou_bg*100:.2f}%  | building: {pseudo_iou_b*100:.2f}%  | mIoU: {pseudo_miou*100:.2f}%')
        print(f'  Δ mIoU vs Pseudo: {(pred_miou - pseudo_miou)*100:+.2f}%')

    print(f'\nImages evaluated: {total_num}')
    print(f'Visualizations saved to: {VIS_DIR}' if SAVE_VIS else '')


if __name__ == '__main__':
    main()
