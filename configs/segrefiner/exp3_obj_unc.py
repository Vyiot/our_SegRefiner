"""
Exp 3: Object + Uncertainty
- M_obj: xóa building theo object uncertainty
- M_unc: nhiễu vùng pixel uncertain (M_unc_region * GT)
- T=6 timesteps
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

work_dir = 'work_dirs/exp3_obj_unc'

model = dict(
    step=6,
    denoise_model=dict(num_timesteps=6),
    diffusion_cfg=dict(
        betas=dict(type='linear', start=0.8, stop=0.0, num_timesteps=6),
        noise_components=dict(use_m_obj=True, use_m_unc=True, use_modify_bnd=False)
    ),
    test_cfg=dict(fine_prob_thr=0.8, max_local_patches=16, nms_iou_thr=0.5)
)

data = dict(
    train=dict(pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=False, with_label=False, with_mask=False, with_seg=True),
        dict(type='LoadOEMCoarseMasks', use_obj=True, use_unc=True, test_mode=False),
        dict(type='RandomCropAll', crop_size=256),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_masks', 'coarse_masks', 'unc_map']),
    ]),
    val=dict(split_file='/home/ubuntu/vy/Denoiser/OEM_v2_Building/val_hard.txt'),
    train_dataloader=dict(samples_per_gpu=16, workers_per_gpu=4),
)

oem_eval = dict(interval=500, num_images=10, save_best=True)
