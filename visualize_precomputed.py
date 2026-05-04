import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Paths
DATA_ROOT = '/home/ubuntu/vy/Denoiser/OpenEarthMap_wo_xBD'
CITY = 'slaskie'
NAME = 'slaskie_32'

img_path = f"{DATA_ROOT}/{CITY}/images/{NAME}.tif"
gt_path = f"{DATA_ROOT}/{CITY}/labels/{NAME}.tif"
unc_path = f"{DATA_ROOT}/{CITY}/uncertainty/{NAME}.npy"
edge_path = f"{DATA_ROOT}/{CITY}/edges/{NAME}.npy"

# Load
img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
gt = cv2.imread(gt_path, 0)
unc = np.load(unc_path)
edge = np.load(edge_path)

# Plot
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.patch.set_facecolor('#0d1117')

axes[0].imshow(img)
axes[0].set_title("Original RGB", color='white')

axes[1].imshow(gt, cmap='gray')
axes[1].set_title("Ground Truth", color='white')

axes[2].imshow(unc, cmap='hot')
axes[2].set_title("Precomputed GMM Uncertainty", color='white')

axes[3].imshow(edge, cmap='gray')
axes[3].set_title("Precomputed Color Canny (Binary)", color='white')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.savefig("/home/ubuntu/vy/Denoiser/SegRefiner/check_precomputed_slaskie_34.png", facecolor='#0d1117')
print("Saved visualization to check_precomputed_slaskie_34.png")
