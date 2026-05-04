"""
segrefiner_oem_base.py
======================
Config cơ sở (base) cho tất cả các thí nghiệm OEM Building.
Kế thừa từ file này, mỗi exp chỉ cần override noise_components.

Dataset: OpenEarthMap_wo_xBD
  - Train: ~3000 ảnh
  - Val:   ~500 ảnh

Object size: 256x256
"""

_base_ = ['../_base_/default_runtime.py']

# ============================================================
# Model
# ============================================================
object_size = 256
task = 'semantic'

model = dict(
    type='SegRefinerSemantic',
    task=task,
    step=12,
    denoise_model=dict(
        type='DenoiseUNet',
        in_channels=4,
        out_channels=1,
        model_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_strides=(16, 32),
        learn_time_embd=True,
        num_timesteps=12,
        channel_mult=(1, 1, 2, 2, 4, 4),
        dropout=0.0),
    diffusion_cfg=dict(
        betas=dict(
            type='linear',
            start=0.9,
            stop=0,
            num_timesteps=12),
        diff_iter=False,
        # [ABLATION] Override trong mỗi file exp con
        noise_components=dict(
            use_obj=False,
            use_bnd=False,
            use_unc=False,
        )
    ),
    test_cfg=dict(
        model_size=1024,
        fine_prob_thr=0.95,
        batch_max=32,
        iou_thr=0.15,
    )
)

# ============================================================
# Image Normalization
# ============================================================
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

# ============================================================
# Pipelines
# ============================================================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=False, with_label=False, with_mask=False, with_seg=True),
    # [NEW] Pipeline đa cấp cho OpenEarthMap
    dict(type='LoadOEMCoarseMasks',
         use_obj=False,
         use_bnd=False,
         use_unc=False,
         obj_unc_threshold=0.3,
         test_mode=False),
    # Lưu global view (1024→256) TRƯỚC khi crop
    dict(type='AddGlobalView', size=object_size),
    # Crop đồng bộ img + masks + unc_map + edge_map cùng 1 vùng ngẫu nhiên 256×256
    dict(type='RandomCropAll', crop_size=object_size),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_masks', 'coarse_masks', 'unc_map', 'edge_map',
                               'global_img', 'global_gt_np', 'global_coarse_np',
                               'global_unc_np', 'global_edge_np']),
]

val_data_root = '/home/ubuntu/vy/Denoiser/OEM_v2_Building'

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=False, with_label=False, with_mask=False, with_seg=True),
    dict(type='LoadOEMCoarseMasks',
         test_mode=True,
         pseudolabel_dir=val_data_root + '/pseudolabels'),
    # dict(type="LoadObjectData"),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['img', 'gt_masks', 'coarse_masks'],
         meta_keys=['filename', 'ori_filename', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                    'img_norm_cfg', 'crop_bbox']),
]

# ============================================================
# Dataset
# ============================================================
data_root = '/home/ubuntu/vy/Denoiser/OpenEarthMap_wo_xBD'

dataset_type = 'OEMBuildingDataset'

data = dict(
    train=dict(
        type=dataset_type,
        data_root=data_root,
        split_file=data_root + '/train.txt',
        pipeline=train_pipeline,
        test_mode=False,
    ),
    val=dict(
        type='OEMv2BuildingDataset',
        data_root=val_data_root,
        split_file=val_data_root + '/val.txt',
        pipeline=val_pipeline,
        test_mode=True,
    ),
    test=dict(
        type='OEMv2BuildingDataset',
        data_root=val_data_root,
        split_file=val_data_root + '/test.txt',
        pipeline=val_pipeline,
        test_mode=True,
    ),
    train_dataloader=dict(samples_per_gpu=2, workers_per_gpu=4),
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=4),
)

# ============================================================
# Optimizer & Scheduler
# ============================================================
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0,
    eps=1e-8,
    betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
opencv_num_threads = 0
mp_start_method = 'spawn'
auto_scale_lr = dict(enable=False, base_batch_size=16)

max_iters = 100000
runner = dict(type='IterBasedRunner', max_iters=max_iters)

lr_config = dict(
    policy='step',
    gamma=0.5,
    by_epoch=False,
    step=[70000, 90000],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=1.0,
    warmup_iters=10)

# ============================================================
# Logging & Evaluation
# ============================================================
log_config = dict(
    interval=25,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False)
    ])

# OEMBuildingEvalHook – tự động được wire vào runner trong mmdet/apis/train.py
# khi phát hiện dataset_type = 'OEMBuildingDataset'.
oem_eval = dict(
    interval=500,
    save_best=True,
)

interval = 1
workflow = [('train', interval)]
# Không lưu checkpoint định kỳ — chỉ giữ best_model.pth do OEMBuildingEvalHook lưu
checkpoint_config = dict(by_epoch=False, interval=0)
