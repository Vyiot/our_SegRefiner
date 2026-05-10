checkpoint_config = dict(interval=0, by_epoch=False)
log_config = dict(
    interval=25, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'spawn'
auto_scale_lr = dict(enable=False, base_batch_size=16)
object_size = 256
task = 'semantic'
model = dict(
    type='SegRefinerSemantic',
    task='semantic',
    step=6,
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
        num_timesteps=6,
        channel_mult=(1, 1, 2, 2, 4, 4),
        dropout=0.0),
    diffusion_cfg=dict(
        betas=dict(type='linear', start=0.8, stop=0.0, num_timesteps=6),
        diff_iter=False,
        noise_components=dict(
            use_m_obj=False, use_m_unc=False, use_modify_bnd=True)),
    test_cfg=dict(
        model_size=1024,
        fine_prob_thr=0.8,
        batch_max=32,
        iou_thr=0.15,
        max_local_patches=16,
        nms_iou_thr=0.5))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=False,
        with_label=False,
        with_mask=False,
        with_seg=True),
    dict(
        type='LoadOEMCoarseMasks',
        use_obj=False,
        use_unc=False,
        obj_unc_threshold=0.3,
        test_mode=False),
    dict(type='RandomCropAll', crop_size=256),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_masks', 'coarse_masks', 'unc_map'])
]
val_data_root = '/home/ubuntu/vy/Denoiser/OEM_v2_Building'
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=False,
        with_label=False,
        with_mask=False,
        with_seg=True),
    dict(
        type='LoadOEMCoarseMasks',
        test_mode=True,
        pseudolabel_dir='/home/ubuntu/vy/Denoiser/OEM_v2_Building/pseudolabels'
    ),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=False),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_masks', 'coarse_masks'],
        meta_keys=[
            'filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
            'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg',
            'crop_bbox'
        ])
]
data_root = '/home/ubuntu/vy/Denoiser/OpenEarthMap_wo_xBD'
dataset_type = 'OEMBuildingDataset'
data = dict(
    train=dict(
        type='OEMBuildingDataset',
        data_root='/home/ubuntu/vy/Denoiser/OpenEarthMap_wo_xBD',
        split_file='/home/ubuntu/vy/Denoiser/OpenEarthMap_wo_xBD/train.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadAnnotations',
                with_bbox=False,
                with_label=False,
                with_mask=False,
                with_seg=True),
            dict(
                type='LoadOEMCoarseMasks',
                use_obj=False,
                use_unc=False,
                obj_unc_threshold=0.3,
                test_mode=False),
            dict(type='RandomCropAll', crop_size=256),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_masks', 'coarse_masks', 'unc_map'])
        ],
        test_mode=False),
    val=dict(
        type='OEMv2BuildingDataset',
        data_root='/home/ubuntu/vy/Denoiser/OEM_v2_Building',
        split_file='/home/ubuntu/vy/Denoiser/OEM_v2_Building/val_hard.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadAnnotations',
                with_bbox=False,
                with_label=False,
                with_mask=False,
                with_seg=True),
            dict(
                type='LoadOEMCoarseMasks',
                test_mode=True,
                pseudolabel_dir=
                '/home/ubuntu/vy/Denoiser/OEM_v2_Building/pseudolabels'),
            dict(type='Resize', img_scale=(1024, 1024), keep_ratio=False),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_masks', 'coarse_masks'],
                meta_keys=[
                    'filename', 'ori_filename', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                    'img_norm_cfg', 'crop_bbox'
                ])
        ],
        test_mode=True),
    test=dict(
        type='OEMv2BuildingDataset',
        data_root='/home/ubuntu/vy/Denoiser/OEM_v2_Building',
        split_file='/home/ubuntu/vy/Denoiser/OEM_v2_Building/test.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadAnnotations',
                with_bbox=False,
                with_label=False,
                with_mask=False,
                with_seg=True),
            dict(
                type='LoadOEMCoarseMasks',
                test_mode=True,
                pseudolabel_dir=
                '/home/ubuntu/vy/Denoiser/OEM_v2_Building/pseudolabels'),
            dict(type='Resize', img_scale=(1024, 1024), keep_ratio=False),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_masks', 'coarse_masks'],
                meta_keys=[
                    'filename', 'ori_filename', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                    'img_norm_cfg', 'crop_bbox'
                ])
        ],
        test_mode=True),
    train_dataloader=dict(samples_per_gpu=16, workers_per_gpu=4),
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=4))
optimizer = dict(
    type='AdamW', lr=0.0001, weight_decay=0, eps=1e-08, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
max_iters = 120000
runner = dict(type='IterBasedRunner', max_iters=120000)
lr_config = dict(
    policy='step',
    gamma=0.5,
    by_epoch=False,
    step=[85000, 110000],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=0.001,
    warmup_iters=500)
oem_eval = dict(interval=500, num_images=10, save_best=True)
interval = 1
work_dir = 'work_dirs1/exp5_bnd'
auto_resume = False
gpu_ids = [0]
