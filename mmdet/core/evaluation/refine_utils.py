import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

def compute_gmm_uncertainty(img_rgb_np: np.ndarray) -> np.ndarray:
    """
    Eq. 1: M_unc = H(GMM(x_rgb)) / log(K)  ∈ [0, 1]
    - W = GMM với K tối ưu chọn bằng BIC (k=2..4)
    - H = entropy của posterior  = -Σ p·log(p)
    - chuẩn hoá bằng log(K)  (max entropy của K class)
    (Clone từ scripts/infer.py)
    """
    H, W, C = img_rgb_np.shape
    pixels   = img_rgb_np.astype(np.float32).reshape(-1, C) / 255.0

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

def nms_patches(candidates, max_patches, iou_threshold):
    """NMS for patches based on uncertainty score."""
    if not candidates:
        return []
    candidates.sort(key=lambda x: x[0], reverse=True)
    keep = []
    while candidates and len(keep) < max_patches:
        best = candidates.pop(0)
        keep.append(best[1:])
        remaining = []
        for cand in candidates:
            iy1, ix1 = max(best[1], cand[1]), max(best[2], cand[2])
            iy2, ix2 = min(best[3], cand[3]), min(best[4], cand[4])
            inter = max(0, iy2 - iy1) * max(0, ix2 - ix1)
            area1 = (best[3] - best[1]) * (best[4] - best[2])
            area2 = (cand[3] - cand[1]) * (cand[4] - cand[2])
            iou = inter / float(area1 + area2 - inter)
            if iou < iou_threshold:
                remaining.append(cand)
        candidates = remaining
    return keep

@torch.no_grad()
def gmm_refine_pipeline(model, img_tensor, img_rgb_np, c_mask_np, device, return_vis=False):
    """
    Refinement pipeline căn chỉnh với train: tìm vùng uncertain → top-48 patches.

    Thay vì sliding window cố định, dùng uncertainty-aware patch selection:
    - Tìm tất cả pixel có unc > 0.3 (giống RandomCropAll trong train)
    - Tạo patch 256×256 centered trên từng pixel uncertain
    - Score = mean unc trong patch → chọn top 48 qua NMS
    """
    H, W = img_tensor.shape[-2:]
    P = 256
    UNC_THRESHOLD = 0.3   # Giống RandomCropAll.unc_map > 0.3 trong train
    MAX_LOCAL_PATCHES = 48  # Top 48 để cover nhiều vùng uncertain hơn
    NMS_IOU_THR = 0.3

    # 1. GMM uncertainty (giống train pipeline)
    unc_map = compute_gmm_uncertainty(img_rgb_np)
    unc_tensor = torch.from_numpy(unc_map).to(device).unsqueeze(0).unsqueeze(0)

    # 2. Tìm patches centered on uncertain pixels — giống RandomCropAll trong train
    #    Lấy tất cả pixel có unc > 0.3, tạo patch 256×256 centered trên đó
    #    Score = mean unc trong patch → NMS giữ top 48
    half_P = P // 2
    unc_mask_bin = (unc_map > UNC_THRESHOLD)
    ys_unc, xs_unc = np.where(unc_mask_bin)

    candidates = []
    if len(ys_unc) > 0:
        # Subsample tâm để không tạo quá nhiều candidates (tối đa 200)
        rng = np.random.RandomState(0)
        n_centers = min(200, len(ys_unc))
        chosen = rng.choice(len(ys_unc), n_centers, replace=False)
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

    c_tensor = torch.from_numpy(c_mask_np.astype(np.float32)).to(device)
    base_mask = c_tensor.unsqueeze(0).unsqueeze(0)

    if not kept:
        res = (base_mask[0, 0] >= 0.5).cpu().numpy().astype(np.uint8)
        return (res, unc_map, []) if return_vis else res

    # 3. Denoising — Cosine window weighted blend
    w_1d = torch.sin(torch.linspace(0, np.pi, P, device=device))
    patch_weight = (w_1d.view(-1, 1) * w_1d.view(1, -1)).view(1, 1, P, P)

    accum_mask = torch.zeros_like(base_mask, dtype=torch.float32)
    accum_weight = torch.zeros_like(base_mask, dtype=torch.float32)

    num_timesteps = getattr(model, 'num_timesteps', 6)
    all_indices = list(range(num_timesteps - 1, -1, -1))

    # Per-step accumulator for vis
    _step_m = {i: torch.zeros_like(base_mask, dtype=torch.float32) for i in all_indices} if return_vis else None
    _step_w = {i: torch.zeros_like(base_mask, dtype=torch.float32) for i in all_indices} if return_vis else None

    for y1, x1, y2, x2 in kept:
        ph, pw = y2 - y1, x2 - x1
        img_patch = img_tensor[:, :, y1:y2, x1:x2]
        mask_patch = base_mask[:, :, y1:y2, x1:x2]
        fp_patch = unc_tensor[:, :, y1:y2, x1:x2]

        if ph < P or pw < P:
            img_patch = F.pad(img_patch, (0, P - pw, 0, P - ph))
            mask_patch = F.pad(mask_patch, (0, P - pw, 0, P - ph))
            fp_patch = F.pad(fp_patch, (0, P - pw, 0, P - ph))

        cur_x = mask_patch.clone()
        cur_fine_probs = fp_patch.clone()

        for i in all_indices:
            t_idx = torch.tensor([i], device=device)
            model_input = torch.cat((img_patch, cur_x), dim=1)
            cur_x, cur_fine_probs = model.p_sample(model_input, cur_fine_probs, t_idx)

            if i == 0:
                vis_val = cur_x.sigmoid().detach()
            else:
                # Bernoulli sampling theo công thức paper (Eq.11)
                fine_map = (torch.rand_like(cur_fine_probs) < cur_fine_probs).float()
                pred_x_start = (cur_x >= 0).float()
                vis_val = (pred_x_start * fine_map + mask_patch * (1 - fine_map)).detach()

            if return_vis:
                pw_ = patch_weight[:, :, :ph, :pw]
                _step_m[i][:, :, y1:y2, x1:x2] += vis_val[:, :, :ph, :pw] * pw_
                _step_w[i][:, :, y1:y2, x1:x2] += pw_

            # Cập nhật cur_x thực tế cho bước tiếp theo
            if i == 0:
                cur_x = cur_x.sigmoid()
            else:
                cur_x = vis_val.clone()

        accum_mask[:, :, y1:y2, x1:x2] += cur_x[:, :, :ph, :pw] * patch_weight[:, :, :ph, :pw]
        accum_weight[:, :, y1:y2, x1:x2] += patch_weight[:, :, :ph, :pw]

    no_patch = (accum_weight == 0)
    accum_mask[no_patch] = base_mask.float()[no_patch]
    accum_weight[no_patch] = 1.0

    result = (accum_mask / (accum_weight + 1e-8))
    final_np = (result[0, 0] >= 0.5).cpu().numpy().astype(np.uint8)

    if not return_vis:
        return final_np

    # Build vis_steps
    vis_steps = []
    for i in all_indices:
        no_p = (_step_w[i] == 0)
        _step_m[i][no_p] = base_mask.float()[no_p]
        _step_w[i][no_p] = 1.0
        step_full = (_step_m[i] / (_step_w[i] + 1e-8))[0, 0].clamp(0, 1).cpu().numpy()
        vis_steps.append((f't={i}(all)', step_full))

    return final_np, unc_map, vis_steps
