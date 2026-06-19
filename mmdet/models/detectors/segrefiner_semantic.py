import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from .segrefiner_base import SegRefiner
from ..builder import DETECTORS, build_head, build_loss
from mmdet.datasets.pipelines.loading import modify_boundary

@DETECTORS.register_module()
class SegRefinerSemantic(SegRefiner):
    
    def get_output_filename(self, img_metas):
        ori_filename = img_metas[0]['ori_filename']
        if 'dis' in ori_filename:
            ori_filename = ori_filename.split('/')
            testset, finename = ori_filename[1], ori_filename[-1]
            output_flie = os.path.join(testset, finename.replace('.jpg', '.png'))
        else:
            output_flie = ori_filename.replace('im.jpg', 'refine.png')
        return output_flie
    

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
    
    def extract_feat(self, img):
        """Directly extract features from the backbone and neck."""
        raise NotImplementedError

    def generate_object_noise(self, gt, t, unc_map=None):
        """Eq. 7: Xóa object C_k nếu U_k^obj > threshold_t (Dynamic Dropout).
        """
        from skimage import measure
        gt_np = (gt[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
        labeled = measure.label(gt_np)
        instances = [ (labeled == i) for i in range(1, labeled.max() + 1) ]

        num_instances = len(instances)
        if num_instances == 0: return torch.zeros_like(gt)

        beta_b = self.betas_cumprod[t]
        M_obj = gt.clone()

        if unc_map is not None:
            # ── [SYNC] Dynamic thresholding dựa trên score_max ────────────
            unc_np     = unc_map[0, 0].cpu().numpy()
            unc_scores = [unc_np[inst].mean() for inst in instances]
            score_max  = max(unc_scores)
            
            # Cbrt schedule (khớp vis script)
            alpha      = (t / max(self.num_timesteps - 1, 1)) ** (1/3)
            threshold  = score_max - alpha * (score_max - score_max / 2.5)

            for i, (inst, u_k) in enumerate(zip(instances, unc_scores)):
                if u_k > threshold:  # U_k^obj > threshold_t -> xóa
                    mask_2d = torch.from_numpy(inst).to(gt.device)
                    M_obj[0, 0, mask_2d] = 0.0
        else:
            # ── Fallback: random selection (không có unc_map) ────────
            # Xóa ngẫu nhiên các object để còn giữ lại tỉ lệ beta_b
            M_obj = torch.zeros_like(gt)
            keep_indices = np.random.choice(
                num_instances,
                max(1, int(num_instances * beta_b)),
                replace=False)
            for idx in keep_indices:
                mask_2d = torch.from_numpy(instances[idx]).to(gt.device)
                M_obj[0, 0, mask_2d] = 1.0

        return M_obj

    def my_modify_boundary(self, image):
        # input: np array of size [H,W] image (uint8)
        # 1. Tạo coarse mask gốc
        coarse_mask = modify_boundary(image)
        # 2. Trích xuất biên (boundary) của coarse mask đó
        coarse_uint8 = (coarse_mask * 255).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(coarse_uint8, kernel, iterations=1)
        eroded = cv2.erode(coarse_uint8, kernel, iterations=1)
        boundary = cv2.subtract(dilated, eroded)
        return (boundary > 127).astype(np.float32)

    # =========================================================
    # [PAPER] Modified Q-Sampling (Eq. 6-11)
    # =========================================================
    def forward_train(self, **kwargs):
        target, x_last, img, current_device = self.get_train_input(**kwargs)
        
        B_total = img.shape[0]
        # Uniform sampling (chuẩn paper)
        t = torch.randint(0, self.num_timesteps, (B_total,), device=current_device)

        x_t = self.q_sample(target, x_last, t, current_device)
        z_t = torch.cat((img, x_t), dim=1)
        
        pred_logits = self.denoise_model(z_t, t) 
        iou_pred = self.cal_iou(target, pred_logits)
        
        losses = dict()
        # Tỷ lệ 2:1 (Lấp đầy : Biên) theo paper
        losses['loss_mask'] = self.loss_mask(pred_logits, target) * 2.0
        losses['loss_texture'] = self.loss_texture(pred_logits, target) * 0.2  # 5.0 * 0.2 = 1.0 → ratio 2:1
        
        losses['iou'] = iou_pred.mean()

        # Log IoU theo từng timestep t
        iou_mean = iou_pred.mean()
        for t_val in range(self.num_timesteps):
            mask = (t == t_val)
            losses[f'iou_t{t_val}'] = iou_pred[mask].mean() if mask.any() else iou_mean

        return losses


    def get_train_input(self, **kwargs):
        """Chỉ lấy Crop view (vùng uncertainly) để train local refinement.
        Bỏ Global view để tập trung tối đa vào độ nét và sửa biên.
        """
        img        = kwargs.get('img', kwargs.get('object_img'))
        gt_masks   = kwargs.get('gt_masks', kwargs.get('object_gt_masks'))
        coarse_masks = kwargs.get('coarse_masks', kwargs.get('object_coarse_masks'))
        unc_map    = kwargs.get('unc_map', kwargs.get('object_unc_map'))

        current_device = img.device
        target = self._bitmapmasks_to_tensor(gt_masks, current_device) if not torch.is_tensor(gt_masks) else gt_masks.to(current_device)
        x_last = self._bitmapmasks_to_tensor(coarse_masks, current_device) if not torch.is_tensor(coarse_masks) else coarse_masks.to(current_device)

        def to_map_tensor(m, name):
            """Chuyển numpy/tensor về (B,1,H,W) float32."""
            if m is None: return None
            if not torch.is_tensor(m):
                m_t = torch.tensor(np.array(m), device=current_device, dtype=torch.float32)
            else:
                m_t = m.to(current_device).float()
            if m_t.dim() == 2: m_t = m_t.unsqueeze(0).unsqueeze(0)
            elif m_t.dim() == 3: m_t = m_t.unsqueeze(1)
            if m_t.shape[-2:] != img.shape[-2:]:
                mode = 'bilinear' if 'unc' in name else 'nearest'
                m_t = F.interpolate(m_t, size=img.shape[-2:], mode=mode,
                                    align_corners=False if mode == 'bilinear' else None)
            return m_t

        self._cur_unc_map = to_map_tensor(unc_map, 'unc')

        return target, x_last, img, current_device


    def q_sample(self, x_start, x_last, t, current_device):
        """
        Modified Q-Sampling theo Eq. 6-11 của bài báo (phiên bản 4).

        Ablation flags (đặt trong config → noise_components):
            use_m_obj      : dùng M_obj làm τ-term trong Eq.11 (Object dynamic dropout)
            use_m_unc      : dùng M_pixel làm (1-τ)-term trong Eq.11 (Pixel flipping)
            use_modify_bnd : thêm M_bnd_t vào I_fused (Boundary dilation noise)
        """
        T = self.num_timesteps
        q_ori_probs = torch.tensor(self.betas_cumprod, device=current_device)
        beta_t_batch = q_ori_probs[t].reshape(-1, 1, 1, 1)  # (B,1,1,1)

        if self._cur_unc_map is None:
            # Fallback khi không có precomputed maps
            sample_noise   = torch.rand_like(x_start)
            transition_map = (sample_noise < beta_t_batch).float()
            return transition_map * x_start + (1 - transition_map) * x_last

        unc_map  = self._cur_unc_map

        results = []
        for b in range(x_start.shape[0]):
            t_val  = t[b].item()
            gt_b   = x_start[b:b+1]
            unc_b  = unc_map[b:b+1]  if b < unc_map.shape[0]  else torch.zeros_like(gt_b)
            beta_b = beta_t_batch[b:b+1]

            # ── Eq. 7: M_obj_t (Dynamic object dropout) ──────────────────────
            M_obj = self.generate_object_noise(gt_b, t_val, unc_map=unc_b)

            # ── Eq. 6: M_unc_t (Pixel uncertainty erosion) ───────────────────
            tau_unc = 0.5
            unc_binary = (unc_b > tau_unc).float()
            n_erode = T - t_val
            if n_erode > 0 and unc_binary.sum() > 0:
                m_unc_t = unc_binary
                for _ in range(n_erode):
                    m_unc_t = 1.0 - F.max_pool2d(1.0 - m_unc_t, kernel_size=3, stride=1, padding=1)
                M_unc = m_unc_t
            else:
                M_unc = unc_binary

            # ── Eq. 5 + Eq. 8: M_bnd_t (Dilated boundary noise) ──────────────
            gt_np = (gt_b[0, 0].cpu().numpy() * 255).astype(np.uint8)
            if gt_np.sum() > 0:
                M_bnd_coarse_np = self.my_modify_boundary(gt_np)
                M_bnd_coarse = torch.from_numpy(M_bnd_coarse_np).to(current_device).float().unsqueeze(0).unsqueeze(0)
            else:
                M_bnd_coarse = torch.zeros_like(gt_b)

            n_bnd = t_val
            if n_bnd > 0 and M_bnd_coarse.sum() > 0:
                M_bnd = M_bnd_coarse
                for _ in range(n_bnd):
                    M_bnd = F.max_pool2d(M_bnd, kernel_size=3, stride=1, padding=1)
            else:
                M_bnd = M_bnd_coarse

            # ── Eq. 9: I_fused = M_unc_t ∪ M_bnd_t ───────────────────────────
            if self.use_modify_bnd:
                I_fused = ((M_bnd > 0.5) | (M_unc > 0.5)).float()
            else:
                I_fused = (M_unc > 0.5).float()

            # ── Eq. 10: M_pixel_t (Pixel flipping) ───────────────────────────
            M_pixel = I_fused * (1.0 - gt_b) + (1.0 - I_fused) * gt_b

            # ── Eq. 11: m_t = τ * M_obj_t + (1 - τ) * M_pixel_t ──────────────
            if not self.use_m_obj and not self.use_m_unc:
                # Fallback nếu tắt cả hai
                m_t = gt_b
            else:
                m_obj_term   = M_obj   if self.use_m_obj   else torch.zeros_like(gt_b)
                m_pixel_term = M_pixel if self.use_m_unc   else torch.zeros_like(gt_b)
                tau = (torch.rand_like(gt_b) < beta_b).float()
                m_t = tau * m_obj_term + (1.0 - tau) * m_pixel_term

            coarse = (m_t >= 0.5).float()
            results.append(coarse)

        return torch.cat(results, dim=0)
