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
from PIL import Image, ImageDraw
import sys

# Add root dir to sys.path
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))
from scripts.infer_test import infer_gmm_refine

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG_FILE = 'configs/segrefiner/exp4_all.py'
CHECKPOINT  = 'work_dirs/exp8_all/best_model3.pth'
OUT_FILE    = 'work_dirs/exp8_all/comparison_3rows_test.png'

# Path to Segformer results (Update this path when you have it)
SEGFORMER_DIR = 'test_oem_raw/test_oem_raw/segformer_results' 

DEVICE = 'cuda:0'
ROOT_DIR = '/home/ubuntu/vy/Denoiser'

SAMPLES = [
    {
        'name': 'aachen_12',
        'img': 'OpenEarthMap_wo_xBD/aachen/images/aachen_12.tif',
        'gt': 'OpenEarthMap_wo_xBD/aachen/labels/aachen_12.tif',
        'pseudo': 'test_oem_raw/test_oem_raw/pseudolabels_binary/aachen_12.tif',
        'baseline': ''
    },
    {
        'name': 'kyoto_27',
        'img': 'OpenEarthMap_wo_xBD/kyoto/images/kyoto_27.tif',
        'gt': 'OpenEarthMap_wo_xBD/kyoto/labels/kyoto_27.tif',
        'pseudo': 'test_oem_raw/test_oem_raw/pseudolabels_binary/kyoto_27.tif',
        'baseline': ''
    },
    {
        'name': 'zanzibar_56',
        'img': 'OpenEarthMap_wo_xBD/zanzibar/images/zanzibar_56.tif',
        'gt': 'OpenEarthMap_wo_xBD/zanzibar/labels/zanzibar_56.tif',
        'pseudo': 'test_oem_raw/test_oem_raw/pseudolabels_binary/zanzibar_56.tif',
        'baseline': ''
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
        img_raw = cv2.imread(osp.join(ROOT_DIR, s['img']))[:, :, ::-1].copy() # BGR to RGB, added copy() to fix negative stride
        gt_raw  = cv2.imread(osp.join(ROOT_DIR, s['gt']), cv2.IMREAD_GRAYSCALE)
        ps_raw  = cv2.imread(osp.join(ROOT_DIR, s['pseudo']), cv2.IMREAD_GRAYSCALE)
        
        # SegRefiner inference
        img_t = torch.from_numpy((img_raw - mean) / std).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
        c_mask = (ps_raw > 0).astype(np.uint8)
        refined_np, _, _ = infer_gmm_refine(model.module if hasattr(model, 'module') else model, 
                                           img_t, img_raw, c_mask, DEVICE)
        
        # Build row panels
        row = [
            to_t(img_raw, is_img=True),
            to_t(ps_raw),
            to_t(refined_np),
            get_diff_map(c_mask, refined_np),
            to_t(gt_raw)
        ]
        all_panels.extend(row)

    # Make Grid with minimal padding
    grid = vutils.make_grid(all_panels, nrow=5, padding=16, pad_value=1.0)
    ndarr = grid.mul(255).clamp(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
    grid_im = Image.fromarray(ndarr)
    
    # Scaled banner height to 120px
    W, H = grid_im.size
    top_margin = 120
    im = Image.new('RGB', (W, H + top_margin), (240, 240, 240)) # Light gray banner
    im.paste(grid_im, (0, top_margin))
    
    # Draw Labels
    draw = ImageDraw.Draw(im)
    from PIL import ImageFont
    # Use confirmed font path on your system
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        # Scaled font size to 100
        font = ImageFont.truetype(font_path, 100)
    except:
        font = None

    labels = ["RGB", "Pseudo(CISC-R)", "Our", "Refinement Map", "GT"]
    S = 1024
    P = 16
    for i, lbl in enumerate(labels):
        try:
            # Precise text centering for modern Pillow
            bbox = draw.textbbox((0, 0), lbl, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            t_offset = bbox[1]
        except:
            w, h, t_offset = 200, 100, 0 # fallback
        
        x_pos = P + i * (S + P) + (S - w) // 2
        y_pos = (top_margin - h) // 2 - t_offset
        draw.text((x_pos, y_pos), lbl, fill=(0, 0, 0), font=font)
    
    os.makedirs(osp.dirname(OUT_FILE), exist_ok=True)
    im.save(OUT_FILE)
    print(f"Done! Visualization saved to: {OUT_FILE}")

if __name__ == '__main__':
    main()
