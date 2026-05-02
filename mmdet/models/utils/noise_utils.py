"""
noise_utils.py
==============
Các hàm tiện ích để tạo Multi-Level Noise cho quá trình huấn luyện.

Ba thành phần:
  A. compute_gmm_uncertainty  -> Bản đồ độ bất định U_score từ GMM trên ảnh RGB
  B. extract_building_instances -> Danh sách các tòa nhà (vùng liên thông) từ GT mask
  C. get_rgb_edges_gpu          -> Bản đồ biên từ ảnh RGB dùng Kornia Canny (GPU)
"""

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from scipy import ndimage
import kornia


# ---------------------------------------------------------------------------
# A. GMM Uncertainty Map
# ---------------------------------------------------------------------------

def compute_gmm_uncertainty(img_rgb: np.ndarray,
                             n_components: int = 3) -> np.ndarray:
    """Tính bản đồ độ bất định (U_score) từ ảnh RGB bằng GMM.

    Args:
        img_rgb (np.ndarray): Ảnh RGB gốc, shape (H, W, 3), dtype uint8 [0,255].
        n_components (int): Số cụm GMM. Mặc định 3 (nhà sáng / nhà tối / nền).

    Returns:
        np.ndarray: Bản đồ entropy shape (H, W), giá trị float [0, 1].
                    Vùng nhập nhằng màu sắc sẽ có giá trị cao.
    """
    H, W, C = img_rgb.shape
    # Chuẩn hóa về [0, 1] trước khi đưa vào GMM
    img_norm = img_rgb.astype(np.float32) / 255.0
    pixels = img_norm.reshape(-1, C)  # (H*W, 3)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        max_iter=50,        # Giới hạn iter để tránh quá chậm khi train
        random_state=0
    )
    gmm.fit(pixels)

    # log_proba: (H*W, n_components)
    log_proba = gmm.predict_proba(pixels)          # xác suất thuộc từng cụm
    # Tính Entropy: H = -sum(p * log(p+eps))
    eps = 1e-8
    entropy = -np.sum(log_proba * np.log(log_proba + eps), axis=1)  # (H*W,)

    # Chuẩn hóa entropy về [0, 1]
    entropy_max = np.log(n_components)             # entropy tối đa
    uncertainty = (entropy / (entropy_max + eps)).reshape(H, W)
    uncertainty = np.clip(uncertainty, 0.0, 1.0).astype(np.float32)
    return uncertainty


# ---------------------------------------------------------------------------
# B. Object Instance Extraction
# ---------------------------------------------------------------------------

def extract_building_instances(gt_mask: np.ndarray):
    """Trích xuất danh sách các tòa nhà (vùng liên thông) từ Ground Truth mask.

    Chú ý: Hàm này chỉ dùng trong quá trình TRAIN, không dùng khi Infer.
    Input mask là Ground Truth (giá trị 0 hoặc 1).

    Args:
        gt_mask (np.ndarray): Binary mask, shape (H, W), dtype uint8, giá trị 0/1.

    Returns:
        list[dict]: Danh sách các dict, mỗi dict gồm:
            - 'mask'  (np.ndarray): Binary mask của tòa nhà đó, shape (H, W)
            - 'area'  (int): Diện tích (số pixel) của tòa nhà
            - 'bbox'  (tuple): (y1, x1, y2, x2) bounding box
    """
    labeled, num_features = ndimage.label(gt_mask.astype(np.int32))
    instances = []
    for label_id in range(1, num_features + 1):
        single_mask = (labeled == label_id).astype(np.uint8)
        area = int(single_mask.sum())
        if area == 0:
            continue
        rows = np.any(single_mask, axis=1)
        cols = np.any(single_mask, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        instances.append({
            'mask': single_mask,
            'area': area,
            'bbox': (int(y1), int(x1), int(y2), int(x2))
        })
    return instances


# ---------------------------------------------------------------------------
# C. RGB Edge Map via Kornia Canny (GPU)
# ---------------------------------------------------------------------------

def get_rgb_edges_gpu(img_rgb_tensor: torch.Tensor,
                       low_threshold: float = 0.1,
                       high_threshold: float = 0.2) -> torch.Tensor:
    """Tìm đường biên của tòa nhà từ ảnh RGB bằng Canny (chạy trực tiếp trên GPU).

    Args:
        img_rgb_tensor (torch.Tensor): Ảnh RGB đã chuẩn hóa về [0,1],
                                       shape (B, 3, H, W) hoặc (3, H, W).
        low_threshold  (float): Ngưỡng thấp Canny. Mặc định 0.1.
        high_threshold (float): Ngưỡng cao Canny. Mặc định 0.2.

    Returns:
        torch.Tensor: Bản đồ biên nhị phân, shape (B, 1, H, W) hoặc (1, H, W),
                      giá trị 0.0 hoặc 1.0.
    """
    squeeze = False
    if img_rgb_tensor.dim() == 3:           # (3, H, W) -> (1, 3, H, W)
        img_rgb_tensor = img_rgb_tensor.unsqueeze(0)
        squeeze = True

    # Chuyển sang ảnh xám để tính Canny
    grayscale = kornia.color.rgb_to_grayscale(img_rgb_tensor)   # (B, 1, H, W)

    # Canny từ Kornia: trả về (magnitude, edges)
    _, edge_map = kornia.filters.canny(
        grayscale,
        low_threshold=low_threshold,
        high_threshold=high_threshold
    )
    edge_map = edge_map.float()   # (B, 1, H, W), giá trị 0.0 hoặc 1.0

    if squeeze:
        edge_map = edge_map.squeeze(0)
    return edge_map
