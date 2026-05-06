"""
oem_eval_hook.py
================
Custom evaluation hook cho OEM Building dataset trong SegRefiner.

Tính 2 loại IoU khi chạy Validation:
  - IoU     : giữa Prediction (sau refine) và Ground Truth
  - Pseudo IoU: giữa Coarse Mask (đầu vào thô) và Ground Truth
    → Dùng làm baseline để biết mô hình đã cải thiện bao nhiêu.

Bảng kết quả hiển thị:
    +------------+-----------+------------+
    | Class      | IoU       | Pseudo IoU |
    +------------+-----------+------------+
    | background | 93.71     | 92.68      |
    | building   | 73.03     | 69.14      |
    | Summary    | mIoU:83.37| mIoU:80.91 |
    +------------+-----------+------------+

Ported & adapted từ SegRefinerDenoiser_Building/mmdet/core/evaluation/oem_eval_hook.py
Điều chỉnh để phù hợp với cấu trúc dataset mới (city-based, không có pseudolabels/).
"""

import os
import os.path as osp

import cv2
import mmcv
import numpy as np
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from mmcv.runner import Hook, HOOKS
from .refine_utils import gmm_refine_pipeline


@HOOKS.register_module()
class OEMBuildingEvalHook(Hook):
    """Periodic evaluation hook cho OEM Building segmentation.

    Chạy inference trên val dataloader mỗi N iterations,
    tính mIoU so với GT labels và Pseudo IoU so với coarse masks.

    Args:
        dataloader: Val DataLoader (test_mode=True).
        data_root (str): Đường dẫn gốc đến OpenEarthMap_wo_xBD.
        interval (int): Số iterations giữa mỗi lần eval.
        save_best (bool): Có lưu checkpoint khi đạt mIoU mới cao nhất.
        log_train_val_size (bool): Log số ảnh train/val khi eval lần đầu.
        train_size (int): Số ảnh train (để log).
        val_size (int): Số ảnh val (để log).
    """

    def __init__(self,
                 dataloader,
                 data_root,
                 interval=5000,
                 num_images=36,
                 save_best=True):
        self.dataloader = dataloader
        self.data_root = data_root
        self.interval = interval
        self.num_images = num_images
        self.save_best = save_best
        self.best_miou = 0.0
        self._first_eval = True

    def after_train_iter(self, runner):
        cur_iter = runner.iter + 1
        if cur_iter % self.interval == 0:
            self._do_evaluate(runner)

    def _do_evaluate(self, runner):
        if self._first_eval:
            val_size = len(self.dataloader.dataset)
            runner.logger.info(f'Validation set: {val_size} images')
            self._first_eval = False

        runner.logger.info(
            f'\n--- Val evaluation at iter {runner.iter + 1} ---')

        model = runner.model
        model.eval()

        # Accumulators cho prediction (refined) và pseudo (coarse)
        pred_ri, pred_ru = 0, 0        # building
        pred_ri_bg, pred_ru_bg = 0, 0  # background
        ps_ri, ps_ru = 0, 0            # building pseudo
        ps_ri_bg, ps_ru_bg = 0, 0      # background pseudo
        total_num = 0
        device = next(model.parameters()).device

        # Helper to denormalize for GMM
        mean = np.array([123.675, 116.28, 103.53])
        std  = np.array([58.395, 57.12, 57.375])

        for data in self.dataloader:
            if total_num >= self.num_images:
                break

            # Lấy tensors từ DataContainer
            img_tensor = data['img'].data[0].to(device)        # (1,3,H,W)
            
            # Xử lý coarse masks (có thể là BitmapMasks hoặc Tensor)
            c_m_raw = data['coarse_masks'].data[0][0]
            if hasattr(c_m_raw, 'masks'):
                c_mask_np = c_m_raw.masks[0]
            else:
                c_mask_np = c_m_raw.numpy()

            # Xử lý gt masks
            gt_m_raw = data['gt_masks'].data[0][0]
            if hasattr(gt_m_raw, 'masks'):
                gt_mask = gt_m_raw.masks[0].astype(np.uint8)
            elif isinstance(gt_m_raw, torch.Tensor):
                gt_mask = gt_m_raw.cpu().numpy().astype(np.uint8)
            else:
                gt_mask = np.array(gt_m_raw).astype(np.uint8)

            # Denormalize to RGB uint8 for GMM
            img_np = img_tensor[0].cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * std + mean).clip(0, 255).astype(np.uint8)

            with torch.no_grad():
                # Chạy pipeline refine mới. Lưu vis cho 5 ảnh đầu.
                do_vis = (total_num < 5)
                if do_vis:
                    refined_np, unc_map, vis_steps = gmm_refine_pipeline(
                        model.module if hasattr(model, 'module') else model,
                        img_tensor, img_np, c_mask_np, device,
                        return_vis=True
                    )
                    self._save_vis_grid(runner, img_np, c_mask_np, unc_map, vis_steps, refined_np, gt_mask, total_num)
                else:
                    refined_np = gmm_refine_pipeline(
                        model.module if hasattr(model, 'module') else model,
                        img_tensor, img_np, c_mask_np, device,
                        return_vis=False
                    )

            # Val IoU: Tính trên ảnh refined
            i, u = self._iou(refined_np, gt_mask)
            pred_ri += i
            pred_ru += u
            i, u = self._iou(1 - refined_np, 1 - gt_mask)
            pred_ri_bg += i
            pred_ru_bg += u

            # Pseudo IoU: Tính trên coarse mask
            i, u = self._iou(c_mask_np, gt_mask)
            ps_ri += i
            ps_ru += u
            i, u = self._iou(1 - c_mask_np, 1 - gt_mask)
            ps_ri_bg += i
            ps_ru_bg += u

            total_num += 1

        # ── Tính mIoU
        pred_iou_b   = pred_ri / max(pred_ru, 1)
        pred_iou_bg  = pred_ri_bg / max(pred_ru_bg, 1)
        pred_miou    = (pred_iou_b + pred_iou_bg) / 2.0

        pseudo_iou_b  = ps_ri / max(ps_ru, 1)
        pseudo_iou_bg = ps_ri_bg / max(ps_ru_bg, 1)
        pseudo_miou   = (pseudo_iou_b + pseudo_iou_bg) / 2.0

        # ── Ghi vào log buffer
        runner.log_buffer.output['val/mIoU'] = pred_miou
        runner.log_buffer.output['val/IoU.building'] = pred_iou_b
        runner.log_buffer.output['val/IoU.background'] = pred_iou_bg
        runner.log_buffer.output['val/Pseudo_mIoU'] = pseudo_miou
        runner.log_buffer.ready = True

        # ── Hiển thị bảng kết quả
        from terminaltables import AsciiTable
        table_data = [
            ['Class', 'Val IoU', 'Pseudo IoU'],
            ['background',
             f'{pred_iou_bg * 100:.2f}',
             f'{pseudo_iou_bg * 100:.2f}'],
            ['building',
             f'{pred_iou_b * 100:.2f}',
             f'{pseudo_iou_b * 100:.2f}'],
            ['Summary',
             f'mIoU: {pred_miou * 100:.2f}',
             f'mIoU: {pseudo_miou * 100:.2f}'],
        ]
        table = AsciiTable(table_data)
        runner.logger.info(
            f'\n{table.table}\n'
            f'Images evaluated: {total_num}')

        # ── Lưu last.pth mỗi lần val (ghi đè cái cũ)
        runner.save_checkpoint(
            runner.work_dir,
            filename_tmpl='last.pth',
            save_optimizer=False)

        # ── Lưu best_model.pth khi đạt mIoU mới cao nhất
        if self.save_best and pred_miou > self.best_miou:
            prev = self.best_miou
            self.best_miou = pred_miou
            save_path = osp.join(runner.work_dir, 'best_model.pth')
            runner.save_checkpoint(
                runner.work_dir,
                filename_tmpl='best_model.pth',
                save_optimizer=False)
            runner.logger.info(
                f'★ New best mIoU: {pred_miou * 100:.2f}% '
                f'(prev: {prev * 100:.2f}%, '
                f'Δ=+{(pred_miou - prev) * 100:.2f}%) '
                f'→ saved to {save_path}')

        model.train()

    def _save_vis_grid(self, runner, img_np, c_mask_np, unc_map, vis_steps, result_np, gt_mask, idx):
        """Lưu grid visualization tương tự infer.py"""
        vis_dir = osp.join(runner.work_dir, 'vis_val', f'iter_{runner.iter + 1}')
        os.makedirs(vis_dir, exist_ok=True)

        P = img_np.shape[0] # Usually 1024

        def to_t(arr):
            t = torch.from_numpy(arr).float()
            if t.ndim == 2: t = t.unsqueeze(0)
            if t.max() > 1: t = t / 255.0
            if t.shape[0] == 1:
                t = t.repeat(3, 1, 1)
            return t

        def img_to_t(img):
            return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        def unc_to_t(unc):
            # Heatmap cho uncertainty
            cm = plt.get_cmap('jet')
            colored = cm(unc)[..., :3]
            return torch.from_numpy(colored).permute(2, 0, 1).float()

        def diff_to_t(coarse, refined):
            # White=Unchanged, Green=Added, Red=Removed
            diff = np.zeros((*coarse.shape, 3), dtype=np.float32)
            diff[(refined == 1) & (coarse == 1)] = [1, 1, 1] # White
            diff[(refined == 1) & (coarse == 0)] = [0, 1, 0] # Green
            diff[(refined == 0) & (coarse == 1)] = [1, 0, 0] # Red
            return torch.from_numpy(diff).permute(2, 0, 1)

        panels = [img_to_t(img_np), to_t(c_mask_np), unc_to_t(unc_map)]
        labels = ['RGB', 'Pseudo', 'Unc']
        
        for lbl, step_arr in vis_steps:
            panels.append(to_t(step_arr))
            labels.append(lbl)
        
        panels += [to_t(result_np), diff_to_t(c_mask_np, result_np), to_t(gt_mask)]
        labels += ['Refined', 'Diff', 'GT']
        
        grid  = vutils.make_grid(panels, nrow=len(panels), padding=8, pad_value=0.5)
        ndarr = grid.mul(255).clamp(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
        im    = Image.fromarray(ndarr)
        
        # Draw labels
        draw = ImageDraw.Draw(im)
        for i, lbl in enumerate(labels):
            # i * (P + padding) + offset
            draw.text((i * (P + 8) + 12, 12), lbl, fill=(255, 0, 0))
            
        im.save(osp.join(vis_dir, f'val_img_{idx}.png'))

    @staticmethod
    def _iou(pred, gt):
        """Tính (intersection, union) cho 2 binary mask."""
        pred = pred.astype(bool)
        gt = gt.astype(bool)
        intersection = np.count_nonzero(pred & gt)
        union = np.count_nonzero(pred | gt)
        return intersection, union

