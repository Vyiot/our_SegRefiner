"""Exp 2: Chỉ Object-level noise"""
_base_ = ['./segrefiner_oem_base.py']
model = dict(diffusion_cfg=dict(noise_components=dict(use_obj=True, use_bnd=False, use_unc=False)))
data = dict(train=dict(pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=False, with_label=False, with_mask=False, with_seg=True),
    dict(type='LoadOEMCoarseMasks', use_obj=True, use_bnd=False, use_unc=False, test_mode=False),
    dict(type='LoadObjectData'),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['object_img', 'object_gt_masks', 'object_coarse_masks']),
]))
