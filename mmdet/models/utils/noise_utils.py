"""
noise_utils.py
==============
Các hàm tiện ích để tạo Multi-Level Noise cho quá trình huấn luyện.

Thành phần:
  A. compute_gmm_uncertainty    -> Bản đồ độ bất định U_score từ GMM trên ảnh RGB
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from scipy import ndimage

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
