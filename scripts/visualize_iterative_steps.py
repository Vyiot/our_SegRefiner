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
from scripts.infer_val import infer_gmm_refine

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG_FILE = 'configs/segrefiner/exp8_all.py'
CHECKPOINT  = 'work_dirs/exp8_all/best_model3.pth'
OUT_FILE    = 'work_dirs/exp8_all/comparison_iterative_steps.png'

DEVICE = 'cuda:0'
ROOT_DIR = '/home/ubuntu/vy/Denoiser'

# Samples for visualization
SAMPLES = [
    {
        'name': 'austin_12',
        'img': 'OEM_v2_Building/images/austin_12.tif',
        'gt': 'OEM_v2_Building/labels/austin_12.tif',
        'pseudo': 'OEM_v2_Building/pseudolabels/austin_12.tif',
    },
    {
        'name': 'melbourne_48',
        'img': 'OEM_v2_Building/images/melbourne_48.tif',
        'gt': 'OEM_v2_Building/labels/melbourne_48.tif',
        'pseudo': 'OEM_v2_Building/pseudolabels/melbourne_48.tif',
    },
    {
        'name': 'ngaoundere_12',
        'img': 'OEM_v2_Building/images/ngaoundere_12.tif',
        'gt': 'OEM_v2_Building/labels/ngaoundere_12.tif',
        'pseudo': 'OEM_v2_Building/pseudolabels/ngaoundere_12.tif',
    }
]

def to_t(arr, size=(1024, 1024), is_img=False):
    if is_img:
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    else:
        if arr.ndim == 2: arr = arr[None]
        t = torch.from_numpy(arr).float()
        
        # Apply 0.5 threshold to make masks clean (binary)
        t = (t > 0.5).float()
            
        t = t.repeat(3, 1, 1)
    return F.interpolate(t.unsqueeze(0), size, mode='bilinear' if is_img else 'nearest')[0]

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
        
        # SegRefiner inference
        img_t = torch.from_numpy((img_raw - mean) / std).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
        c_mask = (ps_raw > 0).astype(np.uint8)
        
        refined_np, _, vis_steps = infer_gmm_refine(model.module if hasattr(model, 'module') else model, 
                                                   img_t, img_raw, c_mask, DEVICE)
        
        # vis_steps is a list of (label, step_full) for t=5,4,3,2,1,0
        step_dict = {label: arr for label, arr in vis_steps}
        
        row = [
            to_t(img_raw, is_img=True),
            to_t(ps_raw),                       # Column 2: Original Pseudo Label
            to_t(step_dict.get('t=5', ps_raw)),  # Column 3: t=5
            to_t(step_dict.get('t=4', ps_raw)),  # Column 4: t=4
            to_t(step_dict.get('t=3', ps_raw)),  # Column 5: t=3
            to_t(step_dict.get('t=2', ps_raw)),  # Column 6: t=2
            to_t(step_dict.get('t=1', ps_raw)),  # Column 7: t=1
            to_t(step_dict.get('t=0', refined_np)), # Column 8: t=0
            to_t(gt_raw)                         # Column 9: GT
        ]
        all_panels.extend(row)

    # Make Grid (9 columns)
    grid = vutils.make_grid(all_panels, nrow=9, padding=16, pad_value=1.0)
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

    labels = ["RGB", "Pseudo", "t=5", "t=4", "t=3", "t=2", "t=1", "t=0", "GT"]
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
    print(f"Done! Iterative visualization saved to: {OUT_FILE}")

if __name__ == '__main__':
    main()
