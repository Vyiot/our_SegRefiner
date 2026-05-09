import os
import os.path as osp
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
import torchvision.utils as vutils
from PIL import Image, ImageDraw, ImageFont
import sys

# Add root dir to sys.path
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))
# Kế thừa từ infer_val
from scripts.infer_val import infer_gmm_refine

# ============================================================
# CONFIGURATION (VAL)
# ============================================================
CONFIG_FILE = 'configs/segrefiner/exp8_all.py'
CHECKPOINT  = 'work_dirs/exp8_all/best_model3.pth'
OUT_FILE    = 'work_dirs/exp8_all/comparison_val_3rows.png'

DEVICE = 'cuda:0'
ROOT_DIR = '/home/ubuntu/vy/Denoiser'

# Các mẫu Validation tiêu biểu từ tập OpenEarthMap
SAMPLES = [
    {
        'name': 'monrovia_23',
        'img': 'OEM_v2_Building/images/monrovia_23.tif',
        'gt': 'OEM_v2_Building/labels/monrovia_23.tif',
        'pseudo': 'OEM_v2_Building/pseudolabels/monrovia_23.tif',
        'baseline': 'infer_selected/oem/monrovia_23.png'
    },
    {
        'name': 'melbourne_50',
        'img': 'OEM_v2_Building/images/melbourne_50.tif',
        'gt': 'OEM_v2_Building/labels/melbourne_50.tif',
        'pseudo': 'OEM_v2_Building/pseudolabels/melbourne_50.tif',
        'baseline': 'infer_selected/oem/melbourne_50.png'
    },
    {
        'name': 'zanzibar_140',
        'img': 'OEM_v2_Building/images/zanzibar_140.tif',
        'gt': 'OEM_v2_Building/labels/zanzibar_140.tif',
        'pseudo': 'OEM_v2_Building/pseudolabels/zanzibar_140.tif',
        'baseline': 'infer_selected/oem/zanzibar_140.png'
    }
]

def to_t(arr, size=(1024, 1024), is_img=False):
    if is_img:
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    else:
        if arr.ndim == 2: arr = arr[None]
        t = torch.from_numpy(arr).float()
        if t.max() > 1: t = t / 255.0
        t = t.repeat(3, 1, 1)
    return F.interpolate(t.unsqueeze(0), size, mode='bilinear' if is_img else 'nearest')[0]

def get_diff_map(pseudo, refined, size=(1024, 1024)):
    p = (pseudo > 0).astype(bool)
    r = (refined > 0).astype(bool)
    diff = np.zeros((*p.shape, 3), dtype=np.float32)
    diff[ p &  r] = [1.0, 1.0, 1.0] # White: Unchanged Building
    diff[ p & ~r] = [1.0, 0.2, 0.2] # Red: Removed
    diff[~p &  r] = [0.2, 0.9, 0.2] # Green: Added
    t = torch.from_numpy(diff).permute(2, 0, 1)
    return F.interpolate(t.unsqueeze(0), size, mode='nearest')[0]

def main():
    cfg = Config.fromfile(CONFIG_FILE)
    model = build_detector(cfg.model).to(DEVICE).eval()
    load_checkpoint(model, CHECKPOINT, map_location='cpu')
    
    mean = np.array([123.675, 116.28, 103.53]).reshape(1, 1, 3)
    std  = np.array([58.395, 57.12, 57.375]).reshape(1, 1, 3)
    
    all_panels = []
    for s in SAMPLES:
        print(f"--> Processing {s['name']}...")
        img_raw = cv2.imread(osp.join(ROOT_DIR, s['img']))[:, :, ::-1].copy() 
        gt_raw  = cv2.imread(osp.join(ROOT_DIR, s['gt']), cv2.IMREAD_GRAYSCALE)
        ps_raw  = cv2.imread(osp.join(ROOT_DIR, s['pseudo']), cv2.IMREAD_GRAYSCALE)
        
        # Load baseline explicitly from SAMPLES
        base_path = s.get('baseline', '')
        base_raw = cv2.imread(osp.join(ROOT_DIR, base_path), cv2.IMREAD_GRAYSCALE) if base_path else None
        if base_raw is None: base_raw = np.zeros_like(ps_raw) if ps_raw is not None else np.zeros((256, 256), dtype=np.uint8)
        
        # SegRefiner inference
        img_t = torch.from_numpy((img_raw - mean) / std).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
        c_mask = (ps_raw > 0).astype(np.uint8)
        
        refined_np, _, _ = infer_gmm_refine(model.module if hasattr(model, 'module') else model, 
                                           img_t, img_raw, c_mask, DEVICE)
        
        row = [
            to_t(img_raw, is_img=True),
            to_t(ps_raw),
            to_t(base_raw),    # Column 3
            to_t(refined_np),   # Column 4
            get_diff_map(c_mask, refined_np),
            to_t(gt_raw)
        ]
        all_panels.extend(row)

    # Make Grid (6 columns)
    grid = vutils.make_grid(all_panels, nrow=6, padding=16, pad_value=1.0)
    ndarr = grid.mul(255).clamp(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
    grid_im = Image.fromarray(ndarr)
    
    # Header banner
    W, H = grid_im.size
    top_margin = 120
    im = Image.new('RGB', (W, H + top_margin), (240, 240, 240))
    im.paste(grid_im, (0, top_margin))
    
    draw = ImageDraw.Draw(im)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        font = ImageFont.truetype(font_path, 100)
    except:
        font = None

    labels = ["RGB", "Pseudo(SegFormer)", "SegRefiner", "Our", "Refinement Map", "GT"]
    S, P = 1024, 16
    for i, lbl in enumerate(labels):
        try:
            bbox = draw.textbbox((0, 0), lbl, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            t_offset = bbox[1]
        except:
            w, h, t_offset = 200, 50, 0
        x_pos = P + i * (S + P) + (S - w) // 2
        y_pos = (top_margin - h) // 2 - t_offset
        draw.text((x_pos, y_pos), lbl, fill=(0, 0, 0), font=font)
    
    os.makedirs(osp.dirname(OUT_FILE), exist_ok=True)
    im.save(OUT_FILE)
    print(f"Done! Validation visualization saved to: {OUT_FILE}")

if __name__ == '__main__':
    main()
