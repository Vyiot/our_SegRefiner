"""Exp 1: SegRefiner gốc – Tất cả noise = False (dùng modify_boundary cũ)"""
_base_ = ['./segrefiner_oem_base.py']

model = dict(diffusion_cfg=dict(noise_components=dict(use_obj=False, use_bnd=False, use_unc=False)))

train_pipeline_override = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=False, with_label=False, with_mask=False, with_seg=True),
    dict(type='LoadOEMCoarseMasks', use_obj=False, use_bnd=False, use_unc=False, test_mode=False),
    dict(type='LoadObjectData'),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['object_img', 'object_gt_masks', 'object_coarse_masks']),
]
data = dict(train=dict(pipeline=train_pipeline_override))
