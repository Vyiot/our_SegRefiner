"""
Exp 5: Object + Boundary
- Giống hệt Exp 8 nhưng tắt use_unc
"""
_base_ = ['./segrefiner_oem_base.py']

max_iters = 120000
runner = dict(type='IterBasedRunner', max_iters=max_iters)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[85000, 110000])

work_dir = 'work_dirs/exp5_obj_bnd'

model = dict(
    step=6,
    denoise_model=dict(num_timesteps=6),
    diffusion_cfg=dict(
        betas=dict(type='linear', start=0.8, stop=0.0, num_timesteps=6),
        noise_components=dict(use_obj=True, use_bnd=True, use_unc=False)
    ),
    test_cfg=dict(fine_prob_thr=0.8, max_local_patches=16, nms_iou_thr=0.5)
)

data = dict(
    train=dict(pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=False, with_label=False, with_mask=False, with_seg=True),
        dict(type='LoadOEMCoarseMasks', use_obj=True, use_bnd=True, use_unc=False, test_mode=False),
        dict(type='RandomCropAll', crop_size=256),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_masks', 'coarse_masks', 'unc_map', 'edge_map']),
    ]),
    train_dataloader=dict(samples_per_gpu=16, workers_per_gpu=4),
)
