import os
import torch
import torch.nn.functional as F
import numpy as np
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
        """Eq. 7: Xóa object C_k nếu U_k^obj > beta_t.
        U_k^obj = mean uncertainty của tất cả pixel trong C_k.
        Nếu không có unc_map, fallback về random selection.
        """
        from skimage import measure
        gt_np = (gt[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
        labeled = measure.label(gt_np)
        instances = [ (labeled == i) for i in range(1, labeled.max() + 1) ]

        num_instances = len(instances)
        if num_instances == 0: return torch.zeros_like(gt)

        beta_b = self.betas_cumprod[t]
        M_obj = torch.zeros_like(gt)

        if unc_map is not None:
            # ── Paper Eq. 7: giữ C_k nếu U_k^obj <= beta_t ──────────
            threshold = beta_b  # Bài báo dùng trực tiếp beta_t
            unc_np = unc_map[0, 0].cpu().numpy()
            kept_any = False
            unc_scores = [unc_np[inst].mean() for inst in instances]
            for i, (inst, u_k) in enumerate(zip(instances, unc_scores)):
                if u_k <= threshold:  # uncertain thấp → giữ
                    mask_2d = torch.from_numpy(inst).to(gt.device)
                    M_obj[0, 0, mask_2d] = 1.0
                    kept_any = True
            # Đảm bảo luôn giữ ít nhất 1 building (chắc chắn nhất)
            if not kept_any:
                best_idx = int(np.argmin(unc_scores))
                mask_2d = torch.from_numpy(instances[best_idx]).to(gt.device)
                M_obj[0, 0, mask_2d] = 1.0
        else:
            # ── Fallback: random selection (không có unc_map) ────────
            keep_indices = np.random.choice(
                num_instances,
                max(1, int(num_instances * beta_b)),
                replace=False)
            for idx in keep_indices:
                mask_2d = torch.from_numpy(instances[idx]).to(gt.device)
                M_obj[0, 0, mask_2d] = 1.0

        return M_obj

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
        Modified Q-Sampling theo Eq. 6-11 của bài báo.

        Ablation flags (đặt trong config → noise_components):
            use_m_obj      : dùng M_obj làm τ-term trong Eq.11
            use_m_unc  : dùng M_unc_region * GT (uncertainty-based) làm (1-τ)-term trong Eq.11
            use_modify_bnd : áp modify_boundary sau Eq.11

        Tất cả intermediate maps luôn được tính đầy đủ để đảm bảo
        gradient flow; flags chỉ kiểm soát đóng góp vào m_t cuối.
        """
        T = self.num_timesteps  # = 6
        q_ori_probs  = torch.tensor(self.betas_cumprod, device=current_device)
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

            # ── Eq. 7: M_obj_t — luôn tính (Eq.7: per-object uncertainty) ─────
            M_obj = self.generate_object_noise(gt_b, t_val, unc_map=unc_b)

            # ── Eq. 6: M_unc_t — luôn tính (Erode unc n=T-t lần) ─────────────
            tau_unc    = 0.5
            unc_binary = (unc_b > tau_unc).float()
            n_erode    = T - t_val
            if n_erode > 0 and unc_binary.sum() > 0:
                neg = 1.0 - unc_binary
                for _ in range(n_erode):
                    neg = F.max_pool2d(neg, kernel_size=3, stride=1, padding=1)
                M_unc_region = 1.0 - neg
            else:
                M_unc_region = unc_binary

            # ── Eq. 9-10: M_pixel_applied = M_unc * GT ───────────────────────
            M_pixel_applied = ((M_unc_region > 0.5) * gt_b).float()

            # ── Eq. 11: gate theo ablation flags ─────────────────────────────
            m_obj_term     = M_obj           if self.use_m_obj     else torch.zeros_like(gt_b)
            m_applied_term = M_pixel_applied if self.use_m_unc else torch.zeros_like(gt_b)
            tau = (torch.rand_like(gt_b) < beta_b).float()
            m_t = tau * m_obj_term + (1 - tau) * m_applied_term

            # ── modify_boundary — scale theo timestep, gate theo use_modify_bnd ──
            # t=0 (β=0.8, sạch): nhẹ  → iou_target=0.90, rates=0.05
            # t=5 (β=0.0, bẩn): mạnh  → iou_target=0.70, rates=0.20
            noise_level = t_val / max(T - 1, 1)
            mb_regional = 0.05 + 0.15 * noise_level
            mb_sample   = 0.05 + 0.15 * noise_level
            mb_iou      = 0.90 - 0.20 * noise_level
            m_t_np = (m_t.detach().cpu().numpy()[0, 0] * 255).astype(np.uint8)
            if self.use_modify_bnd and m_t_np.sum() > 0:
                m_t_mb = modify_boundary(m_t_np,
                                         regional_sample_rate=mb_regional,
                                         sample_rate=mb_sample,
                                         iou_target=mb_iou)
            else:
                m_t_mb = (m_t_np / 255).astype(np.uint8)   # skip / mask rỗng
            m_t = torch.from_numpy(m_t_mb).to(current_device).float().unsqueeze(0).unsqueeze(0)

            results.append(m_t)

        return torch.cat(results, dim=0)
