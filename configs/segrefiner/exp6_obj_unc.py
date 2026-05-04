"""Exp 6: Object + Uncertainty"""
_base_ = ['./segrefiner_oem_base.py']
model = dict(diffusion_cfg=dict(noise_components=dict(use_obj=True, use_bnd=False, use_unc=True)))
data = dict(
    train=dict(pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=False, with_label=False, with_mask=False, with_seg=True),
        dict(type='LoadOEMCoarseMasks', use_obj=True, use_bnd=False, use_unc=True, test_mode=False),
        dict(type='Resize', img_scale=(256, 256), keep_ratio=False),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_masks', 'coarse_masks', 'unc_map', 'edge_map']),
    ]),
    train_dataloader=dict(samples_per_gpu=2)
)
