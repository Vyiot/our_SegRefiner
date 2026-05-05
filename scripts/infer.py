import os
import warnings
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
warnings.filterwarnings('ignore')

"""
infer.py  —  So sánh 3 mode inference với best_model.pth
==========================================================
MODE 0 (stage1_only) : Chỉ Stage 1 global (t=5→1), không Stage 2
MODE 1 (blend)       : Stage 1 + Stage 2 Weighted Blending (hiện tại)
MODE 2 (unc_only)    : Stage 1 + Stage 2 Uncertain-Only Replace (fix đề xuất)

Output: Bảng so sánh mIoU của 3 mode vs Pseudo IoU
"""

import sys
import os.path as osp
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
CONFIG_FILE = 'configs/segrefiner/exp8_all.py'
CHECKPOINT  = 'work_dirs/exp8_all/best_model.pth'
VIS_DIR     = 'work_dirs/exp8_all/vis_compare'
DEVICE      = 'cuda:0'
BATCH_SIZE  = 1
NUM_WORKERS = 4
SAVE_VIS    = True
VIS_MAX     = 10


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


def compute_pseudo_iou(dataset):
    pseudo_dir = osp.join(dataset.data_root, 'pseudolabels')
    label_dir  = osp.join(dataset.data_root, 'labels')
    ri, ru, ri_bg, ru_bg = 0, 0, 0, 0
    for img_name in dataset.img_names:
        basename = osp.splitext(img_name)[0]
        pseudo = cv2.imread(osp.join(pseudo_dir, img_name), cv2.IMREAD_GRAYSCALE)
        gt     = cv2.imread(osp.join(label_dir,  basename + '.tif'), cv2.IMREAD_GRAYSCALE)
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
    return ri / max(ru, 1), ri_bg / max(ru_bg, 1)


# ============================================================
# Core inference — trả về 3 kết quả cùng lúc từ 1 forward pass
# ============================================================
@torch.no_grad()
def infer_all_modes(model_obj, img, c_mask_np, device):
    """
    Chạy đầy đủ pipeline và trả về:
      stage1_res  : Stage 1 only (numpy uint8 H×W)
      blend_res   : Stage 1 + Stage 2 Weighted Blending (numpy uint8 H×W)
      unc_res     : Stage 1 + Stage 2 Uncertain-Only   (numpy uint8 H×W)
    """
    if c_mask_np.sum() <= 128:
        dummy = _to_binary(c_mask_np)
        return dummy, dummy, dummy

    H, W = img.shape[-2:]
    patch_size = 256

    # === Chuẩn bị tensor ===
    c_tensor = torch.from_numpy(c_mask_np.astype(np.float32)).to(device)
    c_4d     = c_tensor.unsqueeze(0).unsqueeze(0)   # (1,1,H,W)

    img_256  = F.interpolate(img, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
    mask_256 = F.interpolate(c_4d, size=(patch_size, patch_size), mode='nearest')

    # === STAGE 1: Global t=5→1 (theo simple_test_semantic gốc) ===
    fine_prob_thr = model_obj.test_cfg.get('fine_prob_thr', 0.8)
    min_commit_prob = 1.0 - fine_prob_thr

    cur_x          = mask_256.clone()
    cur_fine_probs = torch.zeros_like(mask_256)
    global_indices = list(range(1, model_obj.num_timesteps))[::-1]   # [5,4,3,2,1]

    for i in global_indices:
        t = torch.tensor([i], device=device)
        model_input = torch.cat((img_256, cur_x), dim=1)
        cur_x, cur_fine_probs = model_obj.p_sample(model_input, cur_fine_probs, t)

        fine_map    = (cur_fine_probs >= min_commit_prob).float()
        pred_x_start = (cur_x >= 0).float()
        cur_x = pred_x_start * fine_map + mask_256 * (1 - fine_map)

    # Upscale Stage 1 lên 1024
    global_mask_1024  = F.interpolate(cur_x,          size=(H, W), mode='bilinear', align_corners=False)
    fine_probs_1024   = F.interpolate(cur_fine_probs, size=(H, W), mode='bilinear', align_corners=False)

    base_mask = (global_mask_1024 >= 0.5).float()   # binarized Stage 1

    # ── Mode 0: Stage 1 only ─────────────────────────────────
    stage1_res = base_mask[0, 0].cpu().numpy().astype(np.uint8)

    # === STAGE 2: Local t=0 ===
    nms_iou_thr       = model_obj.test_cfg.get('nms_iou_thr', 0.5)
    max_local_patches = model_obj.test_cfg.get('max_local_patches', 16)
    unc_thr           = 1.0 - fine_prob_thr   # pixel uncertain nếu fine_probs < unc_thr

    fp_map = fine_probs_1024[0, 0]   # (H, W)

    # --- Sliding window candidates ---
    stride = patch_size // 2
    ys = list(range(0, max(1, H - patch_size + 1), stride))
    xs = list(range(0, max(1, W - patch_size + 1), stride))
    if ys and ys[-1] < H - patch_size: ys.append(H - patch_size)
    if xs and xs[-1] < W - patch_size: xs.append(W - patch_size)
    if H <= patch_size: ys = [0]
    if W <= patch_size: xs = [0]

    candidates = []
    for y1 in ys:
        for x1 in xs:
            y2 = min(y1 + patch_size, H)
            x2 = min(x1 + patch_size, W)
            patch_fp = fp_map[y1:y2, x1:x2]
            # Tỷ lệ pixel uncertain trong patch
            unc_frac = (patch_fp < unc_thr).float().mean().item()
            if unc_frac <= 0:
                continue
            candidates.append((unc_frac, y1, x1, y2, x2))

    # --- NMS ---
    candidates.sort(key=lambda c: -c[0])
    kept, suppressed = [], set()
    for i, (score, y1, x1, y2, x2) in enumerate(candidates):
        if len(kept) >= max_local_patches:
            break
        if i in suppressed:
            continue
        kept.append((y1, x1, y2, x2))
        for j in range(i + 1, len(candidates)):
            if j in suppressed: continue
            _, y1b, x1b, y2b, x2b = candidates[j]
            iy1, ix1 = max(y1, y1b), max(x1, x1b)
            iy2, ix2 = min(y2, y2b), min(x2, x2b)
            inter = max(0, iy2 - iy1) * max(0, ix2 - ix1)
            union = (y2-y1)*(x2-x1) + (y2b-y1b)*(x2b-x1b) - inter
            if inter / (union + 1e-6) > nms_iou_thr:
                suppressed.add(j)

    # Accumulators cho 2 mode
    # Mode 1: Weighted Blend
    w_1d         = torch.sin(torch.linspace(0, np.pi, patch_size, device=device))
    patch_weight = (w_1d.view(-1, 1) * w_1d.view(1, -1)).view(1, 1, patch_size, patch_size)
    accum_mask   = base_mask.clone()
    accum_weight = torch.ones_like(base_mask)

    # Mode 2: Uncertain-Only Replace
    result_unc = base_mask.clone()

    t0 = torch.tensor([0], device=device)

    for (y1, x1, y2, x2) in kept:
        ph, pw = y2 - y1, x2 - x1
        img_patch   = img[:, :, y1:y2, x1:x2]
        mask_patch  = base_mask[:, :, y1:y2, x1:x2]
        fp_patch    = fine_probs_1024[:, :, y1:y2, x1:x2]

        if ph < patch_size or pw < patch_size:
            img_patch  = F.pad(img_patch,  (0, patch_size-pw, 0, patch_size-ph))
            mask_patch = F.pad(mask_patch, (0, patch_size-pw, 0, patch_size-ph))
            fp_patch   = F.pad(fp_patch,   (0, patch_size-pw, 0, patch_size-ph))

        model_input   = torch.cat((img_patch, mask_patch), dim=1)
        refined_logit, _ = model_obj.p_sample(model_input, fp_patch, t0)

        # ── Mode 1: Weighted Blend (cách cũ) ─────────────────
        refined_prob = refined_logit.sigmoid()
        p_weight     = patch_weight[:, :, :ph, :pw]
        accum_mask[:, :, y1:y2, x1:x2]   += refined_prob[:, :, :ph, :pw] * p_weight
        accum_weight[:, :, y1:y2, x1:x2] += p_weight

        # ── Mode 2: Uncertain-Only Replace (fix) ─────────────
        refined_binary = (refined_logit >= 0).float()
        fp_region      = fp_patch[:, :, :ph, :pw]
        unc_mask       = (fp_region < unc_thr).float()   # 1=uncertain, 0=confident
        refined_out    = refined_binary[:, :, :ph, :pw]
        result_unc[:, :, y1:y2, x1:x2] = (
            refined_out * unc_mask
            + result_unc[:, :, y1:y2, x1:x2] * (1.0 - unc_mask)
        )

    # Finalize Mode 1
    blend_res = ((accum_mask / accum_weight)[0, 0] >= 0.5).cpu().numpy().astype(np.uint8)

    # Finalize Mode 2
    unc_res = result_unc[0, 0].cpu().numpy().astype(np.uint8)

    return stage1_res, blend_res, unc_res


# ============================================================
# Main
# ============================================================
def main():
    cfg = Config.fromfile(CONFIG_FILE)

    # Build val dataset
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

    # Build model
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, CHECKPOINT, map_location='cpu')
    model = model.to(DEVICE)
    model.eval()
    model_obj = model.module if hasattr(model, 'module') else model
    print(f'Loaded: {CHECKPOINT}')
    print(f'num_timesteps = {model_obj.num_timesteps}')
    print(f'betas_cumprod = {model_obj.betas_cumprod}')
    print(f'fine_prob_thr = {model_obj.test_cfg.get("fine_prob_thr", 0.8)}\n')

    # Accumulators cho 3 mode
    metrics = {
        'stage1': [0, 0, 0, 0],   # [ri, ru, ri_bg, ru_bg]
        'blend':  [0, 0, 0, 0],
        'unc':    [0, 0, 0, 0],
    }
    total_num = 0
    vis_count = 0
    label_dir = osp.join(val_dataset.data_root, 'labels')

    for batch_idx, data in enumerate(val_loader):
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

        # Inference 3 mode
        stage1_small, blend_small, unc_small = infer_all_modes(model_obj, img, c_mask_np, DEVICE)

        # Load GT
        gt_mask = None
        if img_name:
            basename = osp.splitext(img_name)[0]
            gt_raw = cv2.imread(osp.join(label_dir, basename + '.tif'), cv2.IMREAD_GRAYSCALE)
            if gt_raw is not None:
                gt_mask = (gt_raw == 1).astype(np.uint8)

        def align_pred(pred_small, gt_mask, crop_bbox):
            """Resize/paste pred về kích thước GT."""
            if gt_mask is None:
                return pred_small
            if crop_bbox is not None:
                h_full, w_full = gt_mask.shape
                y1, x1, y2, x2 = crop_bbox
                crop_h, crop_w = y2 - y1, x2 - x1
                pred_crop = cv2.resize(pred_small, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
                pred_full = np.zeros((h_full, w_full), dtype=np.uint8)
                pred_full[y1:y2, x1:x2] = pred_crop
                return pred_full
            if pred_small.shape != gt_mask.shape:
                return cv2.resize(pred_small, (gt_mask.shape[1], gt_mask.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            return pred_small

        if gt_mask is None:
            gt_mask = _to_binary(c_mask_np)

        pred_s1  = align_pred(stage1_small, gt_mask, crop_bbox)
        pred_bl  = align_pred(blend_small,  gt_mask, crop_bbox)
        pred_unc = align_pred(unc_small,    gt_mask, crop_bbox)

        # Accumulate
        m = metrics['stage1']
        m[0], m[1], m[2], m[3] = _iou_accum(pred_s1,  gt_mask, *m)
        m = metrics['blend']
        m[0], m[1], m[2], m[3] = _iou_accum(pred_bl,  gt_mask, *m)
        m = metrics['unc']
        m[0], m[1], m[2], m[3] = _iou_accum(pred_unc, gt_mask, *m)
        total_num += 1

        # Visualization
        if SAVE_VIS and vis_count < VIS_MAX:
            import torchvision.utils as vutils
            from PIL import Image, ImageDraw

            mean_v = np.array([123.675, 116.28, 103.53]).reshape(3, 1, 1)
            std_v  = np.array([58.395, 57.12, 57.375]).reshape(3, 1, 1)
            img_np = (img[0].cpu().numpy() * std_v + mean_v).clip(0, 255).astype(np.uint8)
            img_np = img_np.transpose(1, 2, 0)   # (H,W,3)

            def to_t(arr, size=256):
                a = _to_binary(arr).astype(np.float32)
                t = torch.from_numpy(a).unsqueeze(0).unsqueeze(0)
                return F.interpolate(t, size=(size, size), mode='nearest')[0].repeat(3, 1, 1)

            def img_to_t(arr_hwc, size=256):
                t = torch.from_numpy(arr_hwc.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
                return F.interpolate(t, size=(size, size), mode='bilinear', align_corners=False)[0]

            panels = [
                img_to_t(img_np),
                to_t(c_mask_np),
                to_t(stage1_small),
                to_t(blend_small),
                to_t(unc_small),
                to_t(gt_mask) if gt_mask is not None else to_t(c_mask_np),
            ]
            labels = ['RGB', 'Pseudo', 'Stage1-Only', 'Blend(cur)', 'UncOnly(fix)', 'GT']

            grid  = vutils.make_grid(panels, nrow=len(panels), padding=4, pad_value=1.0)
            ndarr = grid.mul(255).clamp(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
            im    = Image.fromarray(ndarr)
            draw  = ImageDraw.Draw(im)
            for k, lbl in enumerate(labels):
                draw.text((k * 260 + 6, 6), lbl, fill=(255, 0, 0))

            os.makedirs(VIS_DIR, exist_ok=True)
            name = osp.splitext(img_name)[0] if img_name else f'img{batch_idx:04d}'
            im.save(osp.join(VIS_DIR, f'{name}.png'))
            vis_count += 1

        if (batch_idx + 1) % 50 == 0:
            def running_miou(m):
                b  = m[0] / max(m[1], 1)
                bg = m[2] / max(m[3], 1)
                return (b + bg) / 2 * 100
            print(f'  [{batch_idx+1}/{len(val_loader)}]  '
                  f'S1={running_miou(metrics["stage1"]):.2f}%  '
                  f'Blend={running_miou(metrics["blend"]):.2f}%  '
                  f'UncOnly={running_miou(metrics["unc"]):.2f}%')

    # ── Final metrics ─────────────────────────────────────────
    def calc(m):
        b  = m[0] / max(m[1], 1)
        bg = m[2] / max(m[3], 1)
        return b, bg, (b + bg) / 2

    s1_b,  s1_bg,  s1_m  = calc(metrics['stage1'])
    bl_b,  bl_bg,  bl_m  = calc(metrics['blend'])
    uc_b,  uc_bg,  uc_m  = calc(metrics['unc'])

    pseudo_b, pseudo_bg = compute_pseudo_iou(val_dataset)
    pseudo_m = (pseudo_b + pseudo_bg) / 2

    try:
        from terminaltables import AsciiTable
        table_data = [
            ['Class',      'Stage1-Only',         'Blend (cur)',          'UncOnly (fix)',       'Pseudo'],
            ['background', f'{s1_bg*100:.2f}',    f'{bl_bg*100:.2f}',    f'{uc_bg*100:.2f}',    f'{pseudo_bg*100:.2f}'],
            ['building',   f'{s1_b*100:.2f}',     f'{bl_b*100:.2f}',     f'{uc_b*100:.2f}',     f'{pseudo_b*100:.2f}'],
            ['mIoU',       f'{s1_m*100:.2f}',     f'{bl_m*100:.2f}',     f'{uc_m*100:.2f}',     f'{pseudo_m*100:.2f}'],
        ]
        print('\n' + AsciiTable(table_data).table)
    except ImportError:
        print(f'\n{"="*55}')
        print(f'{"Mode":<18} {"bg IoU":>8} {"bld IoU":>9} {"mIoU":>7}')
        print(f'{"-"*55}')
        print(f'{"Stage1-Only":<18} {s1_bg*100:>8.2f} {s1_b*100:>9.2f} {s1_m*100:>7.2f}')
        print(f'{"Blend (current)":<18} {bl_bg*100:>8.2f} {bl_b*100:>9.2f} {bl_m*100:>7.2f}')
        print(f'{"UncOnly (fix)":<18} {uc_bg*100:>8.2f} {uc_b*100:>9.2f} {uc_m*100:>7.2f}')
        print(f'{"Pseudo (input)":<18} {pseudo_bg*100:>8.2f} {pseudo_b*100:>9.2f} {pseudo_m*100:>7.2f}')
        print(f'{"="*55}')
        print(f'\nΔ Stage1    vs Pseudo: {(s1_m - pseudo_m)*100:+.2f}%')
        print(f'Δ Blend     vs Pseudo: {(bl_m - pseudo_m)*100:+.2f}%')
        print(f'Δ UncOnly   vs Pseudo: {(uc_m - pseudo_m)*100:+.2f}%')

    print(f'\nImages evaluated: {total_num}')
    if SAVE_VIS:
        print(f'Visualizations: {VIS_DIR}')


if __name__ == '__main__':
    main()
