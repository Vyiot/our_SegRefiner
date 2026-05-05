import os
import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage as ndi
from .segrefiner_base import SegRefiner
from ..builder import DETECTORS, build_head, build_loss
from mmcv.ops import nms

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
    
    def simple_test_semantic(self, img_metas, img, coarse_masks, gt_masks=None, **kwargs):
        """2-STAGE GLOBAL-LOCAL INFERENCE.

        Stage 1 - Global (t=5→1):
            Nén ảnh 1024→256, chạy 5 bước diffusion để thu fine_probs
            (bản đồ xác suất cho biết vùng nào đang bị lỗi).

        Stage 2 - Local (t=0):
            Dùng fine_probs tìm vùng lỗi → cắt patch 256×256 từ ảnh
            1024 GỐC (không nén) → chạy bước t=0 để sửa biên sắc nét
            → dán lại vào mask tổng 1024×1024.
        """
        # Reset bộ đếm vis mỗi khi sang đợt Val mới
        cur_iter = kwargs.get('cur_iter', 0)
        if getattr(self, '_last_vis_iter', -1) != cur_iter:
            self._vis_counter = 0
            self._last_vis_iter = cur_iter

        c_mask = coarse_masks[0].masks[0]
        g_mask = gt_masks[0].masks[0] if gt_masks is not None else None

        if coarse_masks[0].masks.sum() <= 128:
            return [(c_mask.copy(), c_mask, g_mask)]

        current_device = img.device
        ori_shape = img_metas[0].get('img_shape', img_metas[0]['ori_shape'])[:2]
        H, W = ori_shape
        patch_size = 256

        # Chuẩn bị coarse mask tensor (1, 1, H, W)
        c_mask_tensor = torch.from_numpy(coarse_masks[0].masks).to(current_device).float()
        if c_mask_tensor.dim() == 2:   c_mask_tensor = c_mask_tensor.unsqueeze(0).unsqueeze(0)
        elif c_mask_tensor.dim() == 3: c_mask_tensor = c_mask_tensor.unsqueeze(1)

        # =====================================================================
        # STAGE 1: GLOBAL (t=5→1) — nén về 256, lấy fine_probs
        # =====================================================================
        img_256   = F.interpolate(img, size=(patch_size, patch_size),
                                  mode='bilinear', align_corners=False)
        mask_256  = F.interpolate(c_mask_tensor, size=(patch_size, patch_size),
                                  mode='nearest')

        cur_x          = mask_256
        cur_fine_probs = torch.zeros_like(mask_256)
        global_indices = list(range(1, self.num_timesteps))[::-1]  # [5,4,3,2,1]
        vis_global_steps = []  # lưu intermediate steps để visualize

        # min_prob = 1 - fine_prob_thr: pixel cần fine_probs >= min_prob để dùng model
        # fine_prob_thr cao → min_prob thấp → model sửa liều hơn (ít cần tự tin hơn)
        for i in global_indices:
            t = torch.tensor([i], device=current_device)
            model_input = torch.cat((img_256, cur_x), dim=1)
            cur_x, cur_fine_probs = self.p_sample(model_input, cur_fine_probs, t)

            # Eq.11: Random sampling theo chuẩn bài báo (giúp Stage 1 bớt "liều")
            fine_map = (torch.rand_like(cur_fine_probs) < cur_fine_probs).float()
            pred_x_start = (cur_x >= 0).float()
            cur_x = pred_x_start * fine_map + mask_256 * (1 - fine_map)

            # Lưu lại step này để vis
            vis_global_steps.append((i, cur_x.squeeze().cpu()))

        # Phóng fine_probs và mask global (logit) lên 1024
        global_mask_1024 = F.interpolate(cur_x, size=(H, W),
                                         mode='bilinear', align_corners=False)
        fine_probs_1024  = F.interpolate(cur_fine_probs, size=(H, W),
                                         mode='bilinear', align_corners=False)

        # =====================================================================
        # STAGE 2: LOCAL (t=0) — NMS chọn patch uncertain, sửa biên
        # =====================================================================
        fine_prob_thr      = self.test_cfg.get('fine_prob_thr', 0.95)
        nms_iou_thr        = self.test_cfg.get('nms_iou_thr', 0.3)
        max_local_patches  = self.test_cfg.get('max_local_patches', 16)

        fp_map = fine_probs_1024[0, 0]  # (H, W)

        # --- Tạo candidate patches bằng sliding window (stride=patch_size//2) ---
        stride = patch_size // 2
        ys = list(range(0, max(1, H - patch_size + 1), stride))
        xs = list(range(0, max(1, W - patch_size + 1), stride))
        if ys and ys[-1] < H - patch_size: ys.append(H - patch_size)
        if xs and xs[-1] < W - patch_size: xs.append(W - patch_size)
        # Xử lý ảnh nhỏ hơn patch_size
        if H <= patch_size: ys = [0]
        if W <= patch_size: xs = [0]

        candidates = []  # (score, y1, x1, y2, x2)
        for y1 in ys:
            for x1 in xs:
                y2 = min(y1 + patch_size, H)
                x2 = min(x1 + patch_size, W)
                patch_fp = fp_map[y1:y2, x1:x2]
                score = 1.0 - patch_fp.mean().item()  # Cao = vùng này model đang phân vân nhất, cần soi kỹ ở Stage 2
                if score > 0:
                    candidates.append((score, y1, x1, y2, x2))

        # --- NMS ---
        candidates.sort(key=lambda c: -c[0])
        kept, suppressed = [], set()
        for i, (score, y1, x1, y2, x2) in enumerate(candidates):
            if len(kept) >= max_local_patches:
                break
            if i in suppressed:
                continue
            kept.append((y1, x1, y2, x2))
            for j in range(i + 1, len(candidates)):
                if j in suppressed: continue
                _, y1b, x1b, y2b, x2b = candidates[j]
                iy1, ix1 = max(y1, y1b), max(x1, x1b)
                iy2, ix2 = min(y2, y2b), min(x2, x2b)
                inter = max(0, iy2 - iy1) * max(0, ix2 - ix1)
                union = (y2-y1)*(x2-x1) + (y2b-y1b)*(x2b-x1b) - inter
                if inter / (union + 1e-6) > nms_iou_thr:
                    suppressed.add(j)

        # --- Xử lý từng patch được chọn bằng Weighted Blending ---
        # Binarize global mask làm nền
        base_mask = (global_mask_1024 >= 0.5).float()
        
        # Tạo trọng số blending (Cosine window) để khử vết cắt vuông
        w_1d = torch.sin(torch.linspace(0, np.pi, patch_size, device=current_device))
        patch_weight = (w_1d.view(-1, 1) * w_1d.view(1, -1)).view(1, 1, patch_size, patch_size)
        
        accum_mask   = base_mask.clone()
        accum_weight = torch.ones_like(base_mask)

        for (y1, x1, y2, x2) in kept:
            ph, pw = y2 - y1, x2 - x1
            img_patch  = img[:, :, y1:y2, x1:x2]
            mask_patch = base_mask[:, :, y1:y2, x1:x2]
            fp_patch   = fine_probs_1024[:, :, y1:y2, x1:x2]

            if ph < patch_size or pw < patch_size:
                img_patch  = F.pad(img_patch,  (0, patch_size-pw, 0, patch_size-ph))
                mask_patch = F.pad(mask_patch, (0, patch_size-pw, 0, patch_size-ph))
                fp_patch   = F.pad(fp_patch,   (0, patch_size-pw, 0, patch_size-ph))

            t0 = torch.tensor([0], device=current_device)
            model_input = torch.cat((img_patch, mask_patch), dim=1)
            
            refined_logit, _ = self.p_sample(model_input, fp_patch, t0)
            refined_prob = refined_logit.sigmoid() # Dùng xác suất để blend cho mượt

            # Cộng dồn vào vùng kết quả kèm trọng số
            p_weight = patch_weight[:, :, :ph, :pw]
            accum_mask[:, :, y1:y2, x1:x2]   += refined_prob[:, :, :ph, :pw] * p_weight
            accum_weight[:, :, y1:y2, x1:x2] += p_weight

        # Kết quả cuối cùng là trung bình có trọng số
        result = (accum_mask / accum_weight)
        res = (result[0, 0] >= 0.5).cpu().numpy().astype(np.uint8)


        # =====================================================================
        # [VIS] Lưu đủ các bước t — 5 sample đầu mỗi đợt Val
        # =====================================================================
        img_idx = self._vis_counter
        if img_idx < 5:
            work_dir = kwargs.get('work_dir', 'work_dirs/exp8_all')
            vis_dir  = os.path.join(work_dir, 'vis')
            os.makedirs(vis_dir, exist_ok=True)

            import torchvision.utils as vutils
            from PIL import Image, ImageDraw

            mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1).to(current_device)
            std  = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1).to(current_device)

            P = patch_size

            # RGB và Pseudo ở 256×256 đầu tiên
            rgb_vis    = (img_256[0] * std + mean).clamp(0, 255) / 255.0  # (3, 256, 256)
            pseudo_vis = mask_256[0].cpu().repeat(3, 1, 1)                # (3, 256, 256)

            vis_list  = [rgb_vis.cpu(), pseudo_vis]
            labels    = ['RGB', 'Pseudo']

            # Các bước global t=5→1
            for (t_val, step_mask) in vis_global_steps:
                vis_list.append(step_mask.unsqueeze(0).repeat(3, 1, 1))
                labels.append(f't={t_val}')

            # Bước local t=0 — Cột 8: Final Result (1024x1024 -> 256x256)
            res_full_binary = (result[0, 0] >= 0.5).float().cpu()
            res_vis = F.interpolate(res_full_binary.unsqueeze(0).unsqueeze(0), 
                                    size=(P, P), mode='nearest').squeeze()
            vis_list.append(res_vis.unsqueeze(0).repeat(3, 1, 1))
            labels.append('t=0(full)')

            # GT
            if gt_masks is not None:
                gt_t = torch.from_numpy(gt_masks[0].masks).float()
                if gt_t.dim() == 2: gt_t = gt_t.unsqueeze(0)
                elif gt_t.dim() == 3: pass
                gt_256 = F.interpolate(gt_t.unsqueeze(0), size=(P, P), mode='nearest').squeeze(0)
                vis_list.append(gt_256.cpu().repeat(3, 1, 1))
                labels.append('GT')

            grid  = vutils.make_grid(vis_list, nrow=len(vis_list), padding=4, pad_value=1.0)
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im    = Image.fromarray(ndarr)
            draw  = ImageDraw.Draw(im)
            for k, label in enumerate(labels):
                draw.text((k * (P + 4) + 6, 6), label, fill=(255, 0, 0))

            img_name = os.path.basename(img_metas[0]['filename']).split('.')[0]
            im.save(f'{vis_dir}/iter{cur_iter}_{img_name}.png')
            self._vis_counter += 1

        return [(res, c_mask, g_mask)]



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
        target, x_last, img, current_device, has_global = self.get_train_input(**kwargs)
        
        B_total = img.shape[0]
        t = torch.zeros(B_total, dtype=torch.long, device=current_device)

        # Uniform sampling (chuẩn paper)
        t = torch.randint(0, self.num_timesteps, (B_total,), device=current_device)

        x_t = self.q_sample(target, x_last, t, current_device)
        z_t = torch.cat((img, x_t), dim=1)
        
        pred_logits = self.denoise_model(z_t, t) 
        iou_pred = self.cal_iou(target, pred_logits)
        
        losses = dict()

        # ── Uncertainty-Weighted Loss ──────────────────────────────────────────
        # Tính trọng số dựa trên Uncertainty Map (GMM)
        # un_weight giúp model tập trung vào các vùng biên và vùng nhãn sai
        unc_weight = 1.0
        if self._cur_unc_map is not None:
            # Đảm bảo shape khớp (B, 1, H, W)
            u_map = self._cur_unc_map
            if u_map.dim() == 3: u_map = u_map.unsqueeze(1)
            unc_weight = 1.0 + 2.0 * u_map.clamp(0.0, 1.0)

        # Tỷ lệ 2:1 (Lấp đầy : Biên) theo paper
        losses['loss_mask'] = self.loss_mask(
            pred_logits, target, weight=unc_weight) * 2.0
        
        losses['loss_texture'] = self._get_texture_loss(
            pred_logits.sigmoid(), target, weight=unc_weight) * 0.2
        
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
        edge_map   = kwargs.get('edge_map', kwargs.get('object_edge_map'))

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
            # Resize nếu cần (thông thường đã khớp 256x256)
            if m_t.shape[-2:] != img.shape[-2:]:
                mode = 'bilinear' if 'unc' in name else 'nearest'
                m_t = F.interpolate(m_t, size=img.shape[-2:], mode=mode,
                                    align_corners=False if mode == 'bilinear' else None)
            return m_t
        self._cur_unc_map  = to_map_tensor(unc_map, 'unc')
        self._cur_edge_map = to_map_tensor(edge_map, 'edge')
        
        return target, x_last, img, current_device, False  # has_global = False


    def q_sample(self, x_start, x_last, t, current_device):
        """
        Modified Q-Sampling theo Eq. 6-11 của bài báo.
        """
        T = self.num_timesteps  # = 6
        q_ori_probs = torch.tensor(self.betas_cumprod, device=current_device)
        beta_t_batch = q_ori_probs[t].reshape(-1, 1, 1, 1)  # (B,1,1,1)

        if self._cur_unc_map is None or self._cur_edge_map is None:
            sample_noise = torch.rand_like(x_start)
            transition_map = (sample_noise < beta_t_batch).float()
            return transition_map * x_start + (1 - transition_map) * x_last

        unc_map  = self._cur_unc_map
        edge_map = self._cur_edge_map

        results = []
        for b in range(x_start.shape[0]):
            t_val = t[b].item()
            gt_b   = x_start[b:b+1]
            unc_b  = unc_map[b:b+1] if b < unc_map.shape[0] else torch.zeros_like(gt_b)
            edge_b = edge_map[b:b+1] if b < edge_map.shape[0] else torch.zeros_like(gt_b)
            beta_b = beta_t_batch[b:b+1]

            # ── Object-level noise (Eq. 7: dùng per-object uncertainty) ─
            M_obj = self.generate_object_noise(gt_b, t_val, unc_map=unc_b) if self.use_obj else gt_b

            # ── Eq. 8: M_bnd_t (Dilate Canny Edge chuẩn bài báo: n_iter = t) ──────
            n_bnd = t_val
            M_bnd = torch.zeros_like(gt_b)
            if self.use_bnd:
                if n_bnd > 0:
                    kernel_bnd = torch.ones(3, 3, device=current_device)
                    # Dilation lặp t_val lần
                    M_bnd = (edge_b > 0.5).float()
                    for _ in range(n_bnd):
                        M_bnd = F.max_pool2d(M_bnd, kernel_size=3, stride=1, padding=1)
                else:
                    # t=0: raw edge, no dilation
                    M_bnd = (edge_b > 0.5).float()

            # ── Eq.6: M_unc_t = Erode(1[M_unc > τ], n_iter=T−t) ────────────
            # t=0 (sạch) → n=6 → erode nhiều → chỉ giữ lõi uncertain cao nhất
            # t=5 (bẩn) → n=1 → erode ít
            # t=T-1=5 (n=1) hoặc nếu n=0 thì giữ nguyên binary
            M_unc_region = torch.zeros_like(gt_b)
            if self.use_unc:
                tau_unc    = 0.5
                unc_binary = (unc_b > tau_unc).float()
                n_erode    = T - t_val   # Eq.6: n_iter = T − t
                if n_erode > 0 and unc_binary.sum() > 0:
                    # Erosion = Dilate(1 − x) rồi lấy 1 − kết quả
                    # Dùng kernel 3×3 lặp n_erode lần (giống cv2.erode)
                    kernel_e  = 3
                    padding_e = 1
                    neg = 1.0 - unc_binary
                    for _ in range(n_erode):
                        neg = F.max_pool2d(neg, kernel_size=kernel_e,
                                          stride=1, padding=padding_e)
                    M_unc_region = 1.0 - neg
                else:
                    M_unc_region = unc_binary

            # ── Eq.10: M_pixel-applied_t — base = M_fine (gt), flip vùng M_sp ─
            # Paper: M_pixel_applied(i,j) = 1-M_fine nếu ∈ M_sp, M_fine otherwise
            M_pixel_applied = gt_b.clone()                                    # base = M_fine
            M_pixel_applied[M_bnd      > 0.5] = 1.0 - gt_b[M_bnd      > 0.5]  # flip M_bnd
            M_pixel_applied[M_unc_region > 0.5] = 1.0 - gt_b[M_unc_region > 0.5]  # flip M_unc

            # ── Eq.11: m_t = τ·M_obj_t + (1−τ)·M_pixel_applied_t ────────────
            # τ ~ Bernoulli(β̄_t): t=0→β=0.8 (mostly M_obj/sạch), t=5→β=0 (toàn M_pixel_applied/bẩn)
            tau = (torch.rand_like(gt_b) < beta_b).float()
            m_t = tau * M_obj + (1 - tau) * M_pixel_applied
            results.append(m_t)

        return torch.cat(results, dim=0)
