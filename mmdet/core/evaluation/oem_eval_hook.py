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
from mmcv.runner import Hook, HOOKS


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
                 save_best=True):
        self.dataloader = dataloader
        self.data_root = data_root
        self.interval = interval
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

        # Accumulators cho prediction (refined)
        pred_ri, pred_ru = 0, 0        # building
        pred_ri_bg, pred_ru_bg = 0, 0  # background
        total_num = 0

        dataset    = self.dataloader.dataset
        label_dir  = osp.join(dataset.data_root, 'labels')

        for data in self.dataloader:
            # Lấy img_meta để lấy tên file và crop_bbox
            try:
                # Correct mmcv DataContainer access: .data[0] = list cho GPU 0, [0] = ảnh đầu tiên
                img_meta = data['img_metas'].data[0][0]
                img_name = img_meta.get('ori_filename', '')
                crop_bbox = img_meta.get('crop_bbox', None) # [y1, x1, y2, x2]
            except Exception:
                img_name = ''
                crop_bbox = None

            with torch.no_grad():
                results = model(return_loss=False, rescale=True, 
                                cur_iter=runner.iter + 1, 
                                work_dir=runner.work_dir, 
                                **data)

            for result in results:
                if result is None:
                    continue

                # Lấy pred và gt từ pipeline (đều đang ở mức crop/resize)
                pred_small, _, gt_small = self._unpack_result(result)
                if pred_small is None or gt_small is None:
                    continue

                # Load GT gốc (1024x1024)
                gt_mask = None
                if img_name:
                    basename = osp.splitext(img_name)[0]
                    gt_raw = cv2.imread(osp.join(label_dir, basename + '.tif'), cv2.IMREAD_GRAYSCALE)
                    if gt_raw is not None:
                        gt_mask = (gt_raw == 1).astype(np.uint8)

                # Nếu có crop_bbox, ghép pred_small (256x256) vào đúng vị trí trên canvas 1024x1024
                if gt_mask is not None and crop_bbox is not None:
                    h_full, w_full = gt_mask.shape
                    y1, x1, y2, x2 = crop_bbox
                    
                    # 1. Resize pred_small về đúng kích thước vùng crop gốc
                    crop_h, crop_w = y2 - y1, x2 - x1
                    pred_crop = cv2.resize(pred_small, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
                    
                    # 2. Tạo canvas trắng và dán vào
                    pred_full = np.zeros((h_full, w_full), dtype=np.uint8)
                    pred_full[y1:y2, x1:x2] = pred_crop
                    
                    pred_mask = pred_full
                elif gt_mask is not None:
                    # [NEW] Nếu KHÔNG có crop_bbox -> Đang chạy toàn ảnh (Full image inference)
                    # Chỉ cần resize pred_small về đúng kích thước GT gốc (1024x1024)
                    if pred_small.shape != gt_mask.shape:
                        pred_mask = cv2.resize(pred_small, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                    else:
                        pred_mask = pred_small
                else:
                    # Fallback cuối cùng
                    pred_mask = pred_small
                    gt_mask = gt_small

                # ── Val IoU: Tính trên ảnh FULL 1024x1024
                i, u = self._iou(pred_mask, gt_mask)
                pred_ri += i
                pred_ru += u
                i, u = self._iou(1 - pred_mask, 1 - gt_mask)
                pred_ri_bg += i
                pred_ru_bg += u

                total_num += 1

        # ── Pseudo IoU: tính trực tiếp từ disk ở kích thước gốc
        pseudo_iou_b, pseudo_iou_bg = self._compute_pseudo_iou_from_disk()
        pseudo_miou = (pseudo_iou_b + pseudo_iou_bg) / 2.0

        # ── Tính mIoU
        pred_iou_b  = pred_ri  / max(pred_ru,  1)
        pred_iou_bg = pred_ri_bg / max(pred_ru_bg, 1)
        pred_miou   = (pred_iou_b + pred_iou_bg) / 2.0

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

    def _compute_pseudo_iou_from_disk(self):
        """Tính Pseudo IoU từ file gốc trên disk (full resolution), không resize.

        Đảm bảo kết quả khớp với baseline 80.91% từ model cũ.
        Returns:
            (iou_building, iou_background): float trong [0, 1]
        """
        dataset = self.dataloader.dataset
        pseudo_dir = osp.join(dataset.data_root, 'pseudolabels')
        label_dir  = osp.join(dataset.data_root, 'labels')

        ri, ru     = 0, 0   # building
        ri_bg, ru_bg = 0, 0  # background

        for img_name in dataset.img_names:
            basename = osp.splitext(img_name)[0]

            pseudo = cv2.imread(osp.join(pseudo_dir, img_name),
                                cv2.IMREAD_GRAYSCALE)
            gt     = cv2.imread(osp.join(label_dir, basename + '.tif'),
                                cv2.IMREAD_GRAYSCALE)
            if pseudo is None or gt is None:
                continue

            # Đồng nhất kích thước nếu lệch
            if pseudo.shape != gt.shape:
                pseudo = cv2.resize(pseudo, (gt.shape[1], gt.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)

            c = (pseudo > 0)          # building predicted
            g = (gt == 1)             # building GT (class 1)

            ri    += np.count_nonzero(c & g)
            ru    += np.count_nonzero(c | g)
            ri_bg += np.count_nonzero(~c & ~g)
            ru_bg += np.count_nonzero(~c | ~g)

        iou_b  = ri    / max(ru,    1)
        iou_bg = ri_bg / max(ru_bg, 1)
        return iou_b, iou_bg

    # ──────────────────────────────────────────────
    # Helper methods
    # ──────────────────────────────────────────────

    @staticmethod
    def _unpack_result(result):
        """Unpack một kết quả inference thành (pred, coarse, gt).

        SegRefiner (test mode) trả về kết quả theo nhiều format khác nhau
        tuỳ vào task. Hàm này chuẩn hoá về binary numpy arrays (H,W).
        """
        try:
            if isinstance(result, (tuple, list)) and len(result) == 3:
                pred, coarse, gt = result
            elif isinstance(result, (tuple, list)) and len(result) == 2:
                # (pred_mask, meta_dict)
                pred = result[0]
                coarse = result[1].get('coarse_mask', None)
                gt = result[1].get('gt_mask', None)
                if coarse is None or gt is None:
                    return None, None, None
            else:
                return None, None, None

            pred = _to_binary(pred)
            coarse = _to_binary(coarse)
            gt = _to_binary(gt)

            # Đảm bảo cùng kích thước
            if pred.shape != gt.shape:
                pred = cv2.resize(
                    pred, (gt.shape[1], gt.shape[0]),
                    interpolation=cv2.INTER_NEAREST)
            if coarse.shape != gt.shape:
                coarse = cv2.resize(
                    coarse, (gt.shape[1], gt.shape[0]),
                    interpolation=cv2.INTER_NEAREST)

            return pred, coarse, gt

        except Exception:
            return None, None, None

    @staticmethod
    def _iou(pred, gt):
        """Tính (intersection, union) cho 2 binary mask."""
        pred = pred.astype(bool)
        gt = gt.astype(bool)
        intersection = np.count_nonzero(pred & gt)
        union = np.count_nonzero(pred | gt)
        return intersection, union


def _to_binary(mask):
    """Chuyển mask về numpy array binary uint8 (H,W), threshold tại 0.5."""
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask = np.squeeze(mask)
    if mask.dtype != np.uint8:
        mask = (mask >= 0.5).astype(np.uint8)
    else:
        mask = (mask > 0).astype(np.uint8)
    return mask
