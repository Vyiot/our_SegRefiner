"""
Exp 8: TẤT CẢ thành phần (Full Pipeline)
- Train lại từ đầu (0 iter)
- Max iters: 120,000
- Sử dụng precomputed maps (GMM + Canny)
"""
_base_ = ['./segrefiner_oem_base.py']

# 1. Train lại từ đầu
resume_from = None
load_from = None

# 2. Cấu hình thời gian huấn luyện
max_iters = 120000
runner = dict(type='IterBasedRunner', max_iters=max_iters)

# 3. Cấu hình Learning Rate (Điều chỉnh theo 120k iter)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[85000, 110000])

# 4. Thư mục lưu kết quả (Giữ nguyên hoặc đổi tên tùy bạn, tôi để mặc định)
work_dir = 'work_dirs/exp8_all'

model = dict(
    step=6,
    denoise_model=dict(
        num_timesteps=6,
    ),
    diffusion_cfg=dict(
        betas=dict(
            type='linear',
            start=0.8,
            stop=0,
            num_timesteps=6),
        noise_components=dict(
            use_obj=True,
            use_bnd=True,
            use_unc=True
        )
    )
)

data = dict(train=dict(pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=False, with_label=False, with_mask=False, with_seg=True),
    dict(type='LoadOEMCoarseMasks', use_obj=True, use_bnd=True, use_unc=True, test_mode=False),
    dict(type='LoadObjectData'),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['object_img', 'object_gt_masks', 'object_coarse_masks', 'object_unc_map', 'object_edge_map']),
]))
