"""
oem_building.py
===============
Dataset classes cho OpenEarthMap Building Segmentation.

Hai class:
  - OEMBuildingDataset : Cấu trúc city-based (OpenEarthMap_wo_xBD)
        <root>/<city>/images/<name>.tif
        <root>/<city>/labels/<name>.tif
  - OEMv2BuildingDataset : Cấu trúc flat (OEM_v2_Building, có pseudolabels)
        <root>/images/<name>.tif
        <root>/labels/<name>.tif
        <root>/pseudolabels/<name>.tif
"""

import os.path as osp
import numpy as np
from .builder import DATASETS
from .pipelines import Compose


def _make_base(data_root, split_file, pipeline, test_mode, tag):
    """Helper khởi tạo chung."""
    with open(split_file, 'r') as f:
        img_names = [l.strip() for l in f if l.strip()]
    print(f'INFO - {"Val" if test_mode else "Train"} set ({tag}): {len(img_names)} images')
    return img_names, Compose(pipeline)


@DATASETS.register_module()
class OEMBuildingDataset:
    """OpenEarthMap với cấu trúc city-based."""

    CLASSES = ('background', 'building')
    PALETTE = [(0, 0, 0), (255, 255, 255)]

    def __init__(self, data_root, split_file, pipeline, test_mode=False, **kwargs):
        self.data_root = data_root
        self.test_mode = test_mode
        self.img_names, self.pipeline = _make_base(
            data_root, split_file, pipeline, test_mode, 'OEM')
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def __len__(self):
        return len(self.img_names)

    def _get_city(self, img_name):
        # Nếu img_name có chứa đường dẫn (vd: city/images/name.tif)
        if '/' in img_name:
            city = img_name.split('/')[0]
            basename = osp.splitext(osp.basename(img_name))[0]
            return city, basename

        # Logic cũ: tìm thư mục khớp với tiền tố của tên file
        basename = osp.splitext(img_name)[0]
        parts = basename.split('_')
        for i in range(len(parts), 0, -1):
            candidate = '_'.join(parts[:i])
            if osp.isdir(osp.join(self.data_root, candidate)):
                return candidate, basename
        raise ValueError(f'Không tìm được thư mục city cho: {img_name}')

    def __getitem__(self, idx):
        img_name = self.img_names[idx] # vd: city/images/name.tif
        city, basename = self._get_city(img_name)
        
        # Đường dẫn seg_map tương ứng: city/labels/name.tif
        seg_map = img_name.replace('/images/', '/labels/')
        
        return self.pipeline(dict(
            img_prefix=self.data_root,
            seg_prefix=self.data_root,
            img_info=dict(filename=img_name),
            ann_info=dict(seg_map=seg_map),
            img_name=img_name,
            city=city,
            seg_fields=[], img_fields=[], mask_fields=[],
        ))


@DATASETS.register_module()
class OEMv2BuildingDataset:
    """OEM_v2_Building với cấu trúc flat và pseudolabels thật."""

    CLASSES = ('background', 'building')
    PALETTE = [(0, 0, 0), (255, 255, 255)]

    def __init__(self, data_root, split_file, pipeline, test_mode=False, **kwargs):
        self.data_root = data_root
        self.test_mode = test_mode
        self.img_names, self.pipeline = _make_base(
            data_root, split_file, pipeline, test_mode, 'OEMv2')
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        basename = osp.splitext(img_name)[0]
        return self.pipeline(dict(
            img_prefix=osp.join(self.data_root, 'images'),
            seg_prefix=osp.join(self.data_root, 'labels'),
            img_info=dict(filename=img_name),
            ann_info=dict(seg_map=basename + '.tif'),
            img_name=img_name,
            seg_fields=[], img_fields=[], mask_fields=[],
        ))
