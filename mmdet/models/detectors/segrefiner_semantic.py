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
        c_mask = coarse_masks[0].masks[0]
        g_mask = gt_masks[0].masks[0] if gt_masks is not None else None

        if coarse_masks[0].masks.sum() <= 128:
            # Mask quá nhỏ → giữ nguyên coarse, không predict zeros (sẽ làm IoU = 0)
            return [(c_mask.copy(), c_mask, g_mask)]
        
        current_device = img.device
        ori_shape = img_metas[0].get('img_shape', img_metas[0]['ori_shape'])[:2]
        model_size = self.test_cfg.get('model_size', 256)
        # [ALGO 2] Quy trình 2 giai đoạn theo đúng giấy báo
        if ori_shape[0] == 1024 and ori_shape[1] == 1024 and model_size == 1024:
            # --- GIAI ĐOẠN 1: Global Refinement ở 256x256 (Các bước T -> 2) ---
            img_256 = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
            
            c_mask_input = coarse_masks[0]
            if not torch.is_tensor(c_mask_input):
                c_mask_tensor = torch.from_numpy(c_mask_input.masks).to(current_device).float()
                if c_mask_tensor.dim() == 3: c_mask_tensor = c_mask_tensor.unsqueeze(1)
            else:
                c_mask_tensor = c_mask_input.to(current_device)
                if c_mask_tensor.dim() == 3: c_mask_tensor = c_mask_tensor.unsqueeze(0)
                elif c_mask_tensor.dim() == 2: c_mask_tensor = c_mask_tensor.unsqueeze(0).unsqueeze(0)
            
            mask_256 = F.interpolate(c_mask_tensor, size=(256, 256), mode='nearest')
            
            indices = list(range(self.num_timesteps))[::-1] # [5, 4, 3, 2, 1, 0]
            global_indices = indices[:-1] # [5, 4, 3, 2, 1] (Bắt đầu từ Bẩn nhất t=5)
            local_indices = [indices[-1]]  # [0] (Bước cuối cùng t=0 - Sạch nhất)

            # [DEBUG VIS] Chuẩn bị list để lưu các bước
            vis_dir = 'debug_vis'
            vis_steps = []
            
            # Giải chuẩn hóa (De-normalize) để ảnh RGB nhìn tự nhiên
            mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1).to(img.device)
            std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1).to(img.device)
            img_show = (img[0] * std + mean) / 255.0 # Đưa về khoảng [0, 1]
            
            vis_steps.append(F.interpolate(img_show.unsqueeze(0), size=(256, 256))[0].cpu())
            vis_steps.append(mask_256[0].cpu().repeat(3, 1, 1))

            # Chạy từng bước một để có thể lưu lại (thay vì chạy cả loop)
            cur_x = mask_256
            cur_fine_probs = torch.zeros_like(mask_256)
            for i in global_indices:
                t = torch.tensor([i], device=current_device)
                model_input = torch.cat((img_256, cur_x), dim=1)
                cur_x, cur_fine_probs = self.p_sample(model_input, cur_fine_probs, t)
                
                # Lưu bước trung gian
                vis_steps.append(cur_x.sigmoid()[0].cpu().repeat(3, 1, 1))
                
                # Algo2 Line 15: dùng M_coarse (mask_256) cố định, KHÔNG phải prev_x
                sample_noise = torch.rand(size=cur_x.shape, device=current_device)
                fine_map = (sample_noise < cur_fine_probs).float()
                pred_x_start = (cur_x >= 0).float()
                cur_x = pred_x_start * fine_map + mask_256 * (1 - fine_map)

            m_2_logits, m_2_fine_probs = cur_x, cur_fine_probs
            
            # Phóng đại kết quả Bước 2 lên lại 1024
            fine_probs_1024 = F.interpolate(m_2_fine_probs, size=(1024, 1024), mode='bilinear', align_corners=False)
            m_2_1024 = F.interpolate(m_2_logits, size=(1024, 1024), mode='bilinear', align_corners=False)

            # --- GIAI ĐOẠN 2: Local Patch Refinement ở 1024x1024 (Chỉ bước 1) ---
            # Truyền m_2_fine_probs (256×256) KHÔNG phải fine_probs_1024 (1024×1024)
            # vì get_local_input tính scale_factor = img_h / model_size = 1024/256 = 4
            # rồi nhân y_c * 4 → nếu y_c đã ở không gian 1024 thì sẽ bị *4 nữa = out of bounds
            patch_imgs, patch_m2_logits, patch_m2_fine_probs, patch_coors = \
                self.get_local_input(img, m_2_1024, m_2_fine_probs, ori_shape)
            
            if patch_imgs is None:
                # m_2_1024 sau bilinear interp của binary mask đã ∈ [0,1]
                # KHÔNG dùng sigmoid (sẽ map sang [0.5, 0.73] và >= 0.5 luôn True)
                final_global = (m_2_1024 >= 0.5).float()
                return [(final_global[0, 0].cpu().numpy(), c_mask, g_mask)]

            batch_max = self.test_cfg.get('batch_max', 32)
            num_ins = len(patch_imgs)
            xs = []
            for idx in range(0, num_ins, batch_max):
                end = min(num_ins, idx + batch_max)
                xs.append((patch_m2_logits[idx:end], patch_imgs[idx:end], patch_m2_fine_probs[idx:end]))

            local_masks_logits, _ = self.p_sample_loop(xs, local_indices, current_device, use_last_step=True)
            
            # m_2_1024 ∈ [0,1] (bilinear interp của binary), KHÔNG dùng sigmoid
            global_mask_binary = (m_2_1024 >= 0.5).float()
            mask = self.paste_local_patch(local_masks_logits, global_mask_binary, patch_coors)
            
            # Chuẩn hóa về 4D (1, 1, H, W) để dùng cho visualization và các bước sau
            mask_4d = mask.unsqueeze(0).unsqueeze(0)
            
            # [DEBUG VIS] Lưu vào work_dir/vis/iterXXX_filename.png
            work_dir = kwargs.get('work_dir', '.')
            cur_iter = kwargs.get('cur_iter', 0)
            vis_dir = os.path.join(work_dir, 'vis')
            
            # Tự động Reset bộ đếm khi sang đợt Val mới
            last_vis_iter = getattr(self, '_last_vis_iter', -1)
            if last_vis_iter != cur_iter:
                self._vis_counter = 0
                self._last_vis_iter = cur_iter

            img_idx = getattr(self, '_vis_counter', 0)
            if img_idx < 5:
                if not os.path.exists(vis_dir):
                    os.makedirs(vis_dir)
                    
                import torchvision.utils as vutils
                from PIL import Image, ImageDraw
                
                # Final result (dùng mask_4d đã chuẩn hóa)
                vis_steps.append(F.interpolate(mask_4d, size=(256, 256))[0].cpu().repeat(3, 1, 1))
                
                # Thêm Ground Truth vào cuối cùng của vis_steps
                if gt_masks is not None:
                    gt_tensor = torch.from_numpy(gt_masks[0].masks).to(current_device).float()
                    if gt_tensor.dim() == 3: gt_tensor = gt_tensor.unsqueeze(1)
                    gt_256 = F.interpolate(gt_tensor, size=(256, 256), mode='nearest')
                    vis_steps.append(gt_256[0].cpu().repeat(3, 1, 1))

                grid = vutils.make_grid(vis_steps, nrow=len(vis_steps), padding=4, pad_value=1.0)
                ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                im = Image.fromarray(ndarr)
                
                draw = ImageDraw.Draw(im)
                # Tên nhãn động: t giảm dần từ (T-1) về 0
                labels = ['RGB', 'Pseudo'] + [f't={i}' for i in global_indices] + ['t=0']
                if gt_masks is not None:
                    labels.append('GT')
                
                for idx, label in enumerate(labels):
                    x_pos = idx * (256 + 4) + 10
                    draw.text((x_pos, 10), label, fill=(255, 0, 0))
                    
                img_name = os.path.basename(img_metas[0]['filename']).split('.')[0]
                save_path = f'{vis_dir}/iter{cur_iter}_{img_name}.png'
                im.save(save_path)
                print(f' >>> Saved labeled visualization to {save_path}')
                self._vis_counter = img_idx + 1
            
            return [(mask_4d[0, 0].cpu().numpy(), c_mask, g_mask)]

        indices = list(range(self.num_timesteps))[::-1]
        global_indices = indices[:-1]
        local_indices = [indices[-1]]

        # global_step
        global_img, global_mask = self._get_global_input(img, coarse_masks, ori_shape, current_device)
        model_size_mask, fine_probs = self.p_sample_loop([(global_mask, global_img, None)], 
                                                        global_indices, 
                                                        current_device, 
                                                        use_last_step=True)
        
        ori_size_mask = F.interpolate(model_size_mask, size=ori_shape)
        ori_size_mask = (ori_size_mask >= 0.5).float()

        # local_step
        patch_imgs, patch_masks, patch_fine_probs, patch_coors = \
            self.get_local_input(img, ori_size_mask, fine_probs, ori_shape)
        if patch_imgs is None:
            return [(ori_size_mask[0, 0].cpu().numpy(), c_mask, g_mask)]
        
        batch_max = self.test_cfg.get('batch_max', 32)
        num_ins = len(patch_imgs)
        if num_ins <= batch_max:
            xs = [(patch_masks, patch_imgs, patch_fine_probs)]
        else:
            xs = []
            for idx in range(0, num_ins, batch_max):
                end = min(num_ins, idx + batch_max)
                xs.append((patch_masks[idx: end], patch_imgs[idx:end], patch_fine_probs[idx:end]))

        local_masks, _ = self.p_sample_loop(xs, 
                                            local_indices, 
                                            patch_imgs.device,
                                            use_last_step=True)
        
        mask = self.paste_local_patch(local_masks, ori_size_mask, patch_coors)
        return [(mask.cpu().numpy(), c_mask, g_mask)]
        # return [(mask.cpu().numpy(), 'test_hr.png')]
    
    def _get_global_input(self, img, coarse_masks, ori_shape, current_device):
        model_size = self.test_cfg.get('model_size', 256)
        coarse_mask = coarse_masks[0].masks[0]
        global_img = F.interpolate(img, size=(model_size, model_size))
        global_mask = torch.tensor(coarse_mask, dtype=torch.float32, device=current_device)
        global_mask = F.interpolate(global_mask.unsqueeze(0).unsqueeze(0), size=(model_size, model_size))
        global_mask = (global_mask >= 0.5).float()
        return global_img, global_mask    
        
    def get_local_input(self, img, ori_size_mask, fine_probs, ori_shape):
        img_h, img_w = ori_shape
        ori_size_fine_probs = F.interpolate(fine_probs, ori_shape)
        fine_prob_thr = self.test_cfg.get('fine_prob_thr', 0.9)
        fine_prob_thr = fine_probs.max().item() * fine_prob_thr
        model_size = self.test_cfg.get('model_size', 256)
        low_cofidence_points = fine_probs < fine_prob_thr
        scores = fine_probs[low_cofidence_points]
        y_c, x_c = torch.where(low_cofidence_points.squeeze(0).squeeze(0))
        scale_factor_y, scale_factor_x = img_h / fine_probs.shape[-2], img_w / fine_probs.shape[-1]
        y_c, x_c = (y_c * scale_factor_y).int(), (x_c * scale_factor_x).int()        
        scores = 1 - scores
        patch_coors = self._get_patch_coors(x_c, y_c, 0, 0, img_w, img_h, model_size, scores, img.device)
        return self.crop_patch(img, ori_size_mask, ori_size_fine_probs, patch_coors)
    
    def _get_patch_coors(self, x_c, y_c, X_1, Y_1, X_2, Y_2, patch_size, scores, device):
        y_1, y_2 = y_c - patch_size/2, y_c + patch_size/2 
        x_1, x_2 = x_c - patch_size/2, x_c + patch_size/2
        invalid_y = y_1 < Y_1
        y_1[invalid_y] = Y_1
        y_2[invalid_y] = Y_1 + patch_size
        invalid_y = y_2 > Y_2
        y_1[invalid_y] = Y_2 - patch_size
        y_2[invalid_y] = Y_2
        invalid_x = x_1 < X_1
        x_1[invalid_x] = X_1
        x_2[invalid_x] = X_1 + patch_size
        invalid_x = x_2 > X_2
        x_1[invalid_x] = X_2 - patch_size
        x_2[invalid_x] = X_2
        
        # Chuyển sang CPU để chạy NMS cho ổn định
        proposals = torch.stack((x_1, y_1, x_2, y_2), dim=-1).cpu().float()
        scores = scores.cpu().float()
        
        if proposals.numel() == 0:
            return torch.empty((0, 4), device=device, dtype=torch.int)
            
        # NMS trên CPU cực nhanh (0.001s), giúp tránh lỗi CUDA mà không làm chậm máy
        patch_coors, _ = nms(proposals, scores, iou_threshold=self.test_cfg.get('iou_thr', 0.2))
        return patch_coors.to(device).int()
    
    def crop_patch(self, img, mask, fine_probs, patch_coors):
        patch_imgs, patch_masks, patch_fine_probs, new_patch_coors = [], [], [], []
        for coor in patch_coors:
            patch_mask = mask[:, :, coor[1]:coor[3], coor[0]:coor[2]]
            if (patch_mask.any()) and (not patch_mask.all()):
                patch_imgs.append(img[:, :, coor[1]:coor[3], coor[0]:coor[2]])
                patch_fine_probs.append(fine_probs[:, :, coor[1]:coor[3], coor[0]:coor[2]])
                patch_masks.append(patch_mask)
                new_patch_coors.append(coor)
        if len(patch_imgs) == 0:
            return None, None, None, None
        patch_imgs = torch.cat(patch_imgs, dim=0)
        patch_masks = torch.cat(patch_masks, dim=0)
        patch_fine_probs = torch.cat(patch_fine_probs, dim=0)
        patch_masks = (patch_masks >= 0.5).float()
        return patch_imgs, patch_masks, patch_fine_probs, new_patch_coors
    
    def paste_local_patch(self, local_masks, mask, patch_coors):
        mask = mask.squeeze(0).squeeze(0)
        refined_mask = torch.zeros_like(mask)
        weight = torch.zeros_like(mask)
        local_masks = local_masks.squeeze(1)
        for local_mask, coor in zip(local_masks, patch_coors):
            refined_mask[coor[1]:coor[3], coor[0]:coor[2]] += local_mask
            weight[coor[1]:coor[3], coor[0]:coor[2]] += 1
        refined_area = (weight > 0).float()
        weight[weight == 0] = 1
        refined_mask = refined_mask / weight
        refined_mask = (refined_mask >= 0.5).float()
        return refined_area * refined_mask + (1 - refined_area) * mask

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
    
    def extract_feat(self, img):
        """Directly extract features from the backbone and neck."""
        raise NotImplementedError

    # =========================================================
    # [PAPER] Modified Q-Sampling (Eq. 6-11)
    # =========================================================
    def get_train_input(self, object_img, object_gt_masks, object_coarse_masks,
                        object_unc_map=None, object_edge_map=None,
                        patch_img=None, patch_gt_masks=None, patch_coarse_masks=None):
        """Override để nhận thêm unc_map và edge_map từ DataLoader."""
        current_device = object_img.device
        img = object_img
        target = self._bitmapmasks_to_tensor(object_gt_masks, current_device)
        x_last = self._bitmapmasks_to_tensor(object_coarse_masks, current_device)

        # Chuyển numpy maps sang tensor (B, 1, H, W)
        if object_unc_map is not None:
            if not torch.is_tensor(object_unc_map):
                self._cur_unc_map = torch.tensor(object_unc_map, device=current_device, dtype=torch.float32)
            else:
                self._cur_unc_map = object_unc_map.to(current_device).float()
            if self._cur_unc_map.dim() == 3:
                self._cur_unc_map = self._cur_unc_map.unsqueeze(1)  # (B,1,H,W)
        else:
            self._cur_unc_map = None

        if object_edge_map is not None:
            if not torch.is_tensor(object_edge_map):
                self._cur_edge_map = torch.tensor(object_edge_map, device=current_device, dtype=torch.float32)
            else:
                self._cur_edge_map = object_edge_map.to(current_device).float()
            if self._cur_edge_map.dim() == 3:
                self._cur_edge_map = self._cur_edge_map.unsqueeze(1)  # (B,1,H,W)
        else:
            self._cur_edge_map = None

        if patch_img is not None:
            img = torch.cat((img, patch_img), dim=0)
            target = torch.cat((target, self._bitmapmasks_to_tensor(patch_gt_masks, current_device)), dim=0)
            x_last = torch.cat((x_last, self._bitmapmasks_to_tensor(patch_coarse_masks, current_device)), dim=0)

        return target, x_last, img, current_device

    def q_sample(self, x_start, x_last, t, current_device):
        """
        Modified Q-Sampling theo Eq. 6-11 của bài báo.
        
        x_start: GT mask  (B, 1, H, W)
        x_last:  Coarse mask ban đầu (B, 1, H, W) - dùng làm M_fine
        t:       Timestep tensor (B,)
        
        Nếu không có unc_map / edge_map, fallback về q_sample gốc.
        """
        T = self.num_timesteps  # = 6
        # Lấy beta_t cho từng sample trong batch (scalar đầu tiên)
        q_ori_probs = torch.tensor(self.betas_cumprod, device=current_device)
        beta_t_batch = q_ori_probs[t].reshape(-1, 1, 1, 1)  # (B,1,1,1)

        # Nếu không có maps, dùng q_sample gốc đơn giản
        if self._cur_unc_map is None or self._cur_edge_map is None:
            sample_noise = torch.rand_like(x_start)
            transition_map = (sample_noise < beta_t_batch).float()
            return transition_map * x_start + (1 - transition_map) * x_last

        unc_map  = self._cur_unc_map   # (B,1,H,W)
        edge_map = self._cur_edge_map  # (B,1,H,W)
        # M_fine = GT (x_start) lúc Train, vì mô hình học từ GT sạch rồi add noise

        results = []
        for b in range(x_start.shape[0]):
            t_val = t[b].item()  # timestep của sample này
            gt_b   = x_start[b:b+1]   # (1,1,H,W) - M_fine = GT trong training
            unc_b  = unc_map[b:b+1] if b < unc_map.shape[0] else torch.zeros_like(gt_b)
            edge_b = edge_map[b:b+1] if b < edge_map.shape[0] else torch.zeros_like(gt_b)
            beta_b = beta_t_batch[b:b+1]

            # ── Eq. 2-4 & 7: Object-level noise (M_obj_coarse_t) ───────────
            # Eq. 2: FindObject(M_fine) = FindObject(GT) → tìm từng tòa nhà riêng lẻ
            # Eq. 3: U_k^obj = mean(M_unc_coarse) trên C_k của từng tòa nhà
            # Eq. 7: M_obj_coarse_t(i,j) = 0 nếu (i,j)∈C_k VÀ U_k^obj > β̄_t
            #                             = M_fine(i,j) = GT(i,j) nếu không
            M_obj = gt_b.clone()  # Bắt đầu từ M_fine = GT
            if self.use_obj:
                gt_np    = (gt_b[0, 0].cpu().numpy() > 0.5).astype(np.int32)
                unc_np   = unc_b[0, 0].cpu().numpy()
                beta_val = beta_b.squeeze().item()  # β̄_t
                labeled, num_buildings = ndi.label(gt_np)
                if num_buildings > 0:
                    M_obj_np = gt_np.copy().astype(np.float32)  # Bắt đầu từ GT
                    for k in range(1, num_buildings + 1):
                        mask_k = (labeled == k)       # Eq. 2: Pixels thuộc tòa nhà k
                        U_k = unc_np[mask_k].mean()   # Eq. 3: U_k^obj
                        if U_k > beta_val:             # Eq. 7: xóa tòa nhà nếu quá uncertain
                            M_obj_np[mask_k] = 0.0
                    M_obj = torch.from_numpy(M_obj_np).to(gt_b.device).unsqueeze(0).unsqueeze(0)

            # ── Eq. 8: Boundary-level noise (M_bnd_coarse_t) ───────────────
            # Dilate M_bnd_coarse (Canny RGB) với n_iter = t lần
            # t=0 → n_iter=0 → không dilate, giữ nguyên M_bnd_coarse gốc
            # t>0 → dilate t lần (approx bằng max_pool2d kernel 2t+1)
            M_bnd = (edge_b > 0.5).float() if self.use_bnd else torch.zeros_like(gt_b)
            if self.use_bnd and t_val > 0:
                # max_pool2d kernel (2t+1) ≈ t iterations của morphological dilation 3×3
                kernel  = 2 * t_val + 1
                padding = t_val
                M_bnd = F.max_pool2d(edge_b, kernel_size=kernel, stride=1, padding=padding)
                M_bnd = (M_bnd > 0.5).float()

            # ── Eq. 6: Uncertainty-level noise (M_unc_coarse_t) ────────────
            # Erode(1[M_unc_coarse > τ_unc], n_iter = T-t)
            # t=5 (noisy): T-t=1 → erosion nhẹ, giữ nhiều vùng uncertain
            # t=0 (sạch):  T-t=6 → erosion mạnh, chỉ còn lõi uncertain
            M_unc_region = torch.zeros_like(gt_b)
            if self.use_unc:
                tau_unc    = 0.3
                unc_binary = (unc_b > tau_unc).float()   # 1[M_unc_coarse > τ_unc]
                n_erode    = T - t_val
                if n_erode > 0 and unc_binary.any():
                    kernel_e  = 2 * n_erode + 1
                    padding_e = n_erode
                    # Erosion = ~Dilate(~unc_binary)
                    neg          = 1.0 - unc_binary
                    dilated_neg  = F.max_pool2d(neg, kernel_size=kernel_e, stride=1, padding=padding_e)
                    M_unc_region = 1.0 - dilated_neg
                else:
                    # n_erode = 0 (tức t = T): không erode, giữ nguyên
                    M_unc_region = unc_binary

            # ── Eq. 9: Super-pixel noise ──────────────────────────────────
            # M_Sp_coarse_t = M_bnd_coarse_t ∪ M_unc_coarse_t
            M_Sp = ((M_bnd + M_unc_region) > 0).float()

            # ── Eq. 10: Pixel-applied coarse mask ─────────────────────────
            # M_pixel_applied_t(i,j) = 1-M_fine(i,j) nếu M_Sp=1
            #                        = M_fine(i,j)   nếu M_Sp=0
            M_pixel_applied = torch.where(M_Sp > 0.5,
                                          1.0 - gt_b,   # Lật pixel trong vùng Sp
                                          gt_b)         # Giữ GT bên ngoài vùng Sp

            # ── Eq. 11: Final Q-sampling ──────────────────────────────────
            # m_t = τ * M_obj_coarse_t + (1-τ) * M_pixel_applied_t
            # τ^{i,j} = 1[Uniform(0,1) < β̄_t]
            tau = (torch.rand_like(gt_b) < beta_b).float()
            m_t = tau * M_obj + (1 - tau) * M_pixel_applied
            results.append(m_t)

        return torch.cat(results, dim=0)
