"""
Visualize RGB | GT | Unc map | M_unc | Edge map cho nhiều ảnh cùng lúc.

Cấu trúc data:
    data_root/
        city/
            images/      basename.tif  (hoặc .png / .jpg)
            labels/      basename.png
            uncertainty/ basename.npy
            edges/       basename.npy

Usage:
    python vis_unc_obj.py --data_root /path/to/data --n 8 --out vis_unc_obj.png
    python vis_unc_obj.py --data_root /path/to/data --city austin --n 5
"""

import argparse, glob, os, random
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage import measure

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', required=True)
parser.add_argument('--city',   default=None,  help='Chỉ lấy 1 thành phố cụ thể')
parser.add_argument('--n',      type=int, default=6, help='Số ảnh lấy (random)')
parser.add_argument('--tau',    type=float, default=0.5)
parser.add_argument('--out',    default='vis_unc_obj.png')
parser.add_argument('--seed',   type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)

# ── Tìm tất cả ảnh có đủ unc + gt ──────────────────────────────────────────
cities = [args.city] if args.city else os.listdir(args.data_root)
samples = []
for city in sorted(cities):
    img_dir  = os.path.join(args.data_root, city, 'images')
    unc_dir  = os.path.join(args.data_root, city, 'uncertainty')
    edge_dir = os.path.join(args.data_root, city, 'edges')
    gt_dir   = os.path.join(args.data_root, city, 'labels')
    if not os.path.isdir(img_dir): continue
    for img_path in glob.glob(os.path.join(img_dir, '*')):
        base = os.path.splitext(os.path.basename(img_path))[0]
        unc_path  = os.path.join(unc_dir,  base + '.npy')
        edge_path = os.path.join(edge_dir, base + '.npy')
        gt_path = None
        for ext in ('.png', '.tif', '.jpg'):
            p = os.path.join(gt_dir, base + ext)
            if os.path.exists(p):
                gt_path = p; break
        if os.path.exists(unc_path) and gt_path:
            samples.append((img_path, gt_path, unc_path, edge_path))

print(f'Tổng ảnh có đủ dữ liệu: {len(samples)}')
if not samples:
    print('Không tìm thấy ảnh nào. Kiểm tra lại --data_root và cấu trúc thư mục.')
    exit(1)

chosen = random.sample(samples, min(args.n, len(samples)))

# ── Hàm vẽ 1 hàng ─────────────────────────────────────────────────────────
def draw_row(axes, img_path, gt_path, unc_path, edge_path):
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gt_raw  = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = (gt_raw > 0).astype(np.uint8)   # OEM: building=1, background=0

    unc_map = np.load(unc_path).astype(np.float32)

    H, W = gt_mask.shape
    unc_map = cv2.resize(unc_map, (W, H), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.resize(img_rgb, (W, H))

    # M_unc binary
    M_unc = (unc_map > args.tau).astype(np.float32)

    # Edge map
    if os.path.exists(edge_path):
        edge_map = np.load(edge_path).astype(np.float32)
        edge_map = cv2.resize(edge_map, (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        edge_map = np.zeros((H, W), dtype=np.float32)

    # Vẽ 5 cột: RGB | GT | Unc map | M_unc | Edge
    axes[0].imshow(img_rgb);                               axes[0].axis('off')
    axes[1].imshow(gt_mask, cmap='gray', vmin=0, vmax=1);  axes[1].axis('off')
    axes[2].imshow(unc_map, cmap='hot',  vmin=0, vmax=1);  axes[2].axis('off')
    axes[3].imshow(M_unc,   cmap='gray', vmin=0, vmax=1);  axes[3].axis('off')
    axes[4].imshow(edge_map,cmap='gray', vmin=0, vmax=1);  axes[4].axis('off')

    name = os.path.basename(img_path)
    axes[0].set_ylabel(name, fontsize=7, rotation=0, ha='right', va='center', labelpad=55)

# ── Tạo figure ───────────────────────────────────────────────────────────────
N = len(chosen)
fig, axes = plt.subplots(N, 5, figsize=(22, 4 * N))
if N == 1: axes = axes[np.newaxis, :]

col_titles = ['RGB gốc', 'GT mask', 'Unc map (raw)',
              f'M_unc  (unc > {args.tau})', 'Edge map']
for j, title in enumerate(col_titles):
    axes[0, j].set_title(title, fontsize=9, fontweight='bold')

for i, (img_p, gt_p, unc_p, edge_p) in enumerate(chosen):
    draw_row(axes[i], img_p, gt_p, unc_p, edge_p)

plt.tight_layout()
plt.savefig(args.out, dpi=130, bbox_inches='tight')
print(f'Saved → {args.out}  ({N} ảnh)')
