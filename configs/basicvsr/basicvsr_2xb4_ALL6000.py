_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/basicvsr_test_config.py'
]

load_batch_size = 8
load_input_frames = 10
name_input_frames = 2 * load_input_frames  # doubled because flip augmentation doubles the data
experiment_name = f'basicvsr-c64b30_1x{name_input_frames}xb{load_batch_size}_ALL6000'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs'

scale = 4
# load the checkpoint for faster adaptation

# model settings
model = dict(
    type='BasicVSR',
    generator=dict(
        type='BasicVSRNet',
        mid_channels=64,
        num_blocks=30,
        spynet_pretrained='ckpt/spynet_20210409-c6c1bd09.pth'),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    # train_cfg=dict(fix_iter=5000),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

train_pipeline = [
    dict(type='GenerateSegmentIndices',
         interval_list=[1],
         filename_tmpl='{:05d}.jpg'
         ),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(type='FixedCrop', keys=['img', 'gt'], crop_size=(256, 256)),
    dict(type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['img', 'gt'], transpose_ratio=0.5),
    dict(type='MirrorSequence', keys=['img', 'gt']),  # extend the sequence (x1, ..., xN, xN, ..., x1).
    # dict(
    #     type='UnsharpMasking',
    #     keys=['gt'],
    #     kernel_size=51,
    #     sigma=0,
    #     weight=0.5,
    #     threshold=10),
    # removed degradation
    # randomly downscale to 64x64 to train super-resolution up to 256x256
    dict(type='RandomResize',
         params=dict(
             target_size=(64, 64),
             resize_opt=['bilinear', 'area', 'bicubic'],
             resize_prob=[1 / 3., 1 / 3., 1 / 3.]),
         keys=['img']),
    dict(type='Clip', keys=['img']),  # handle inputs of inconsistent sizes; both image size and the number of input frames may matter
    dict(type='PackInputs')
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='PackInputs')
]

data_root = 'data'

train_dataloader = dict(
    num_workers=10,
    batch_size=load_batch_size,  # must be even or 1
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='ALL6000', task_name='vsr'),
        data_root=f'{data_root}/ALL6000x5',
        data_prefix=dict(img='IMG', gt='GT'),
        depth=1,
        num_input_frames=load_input_frames,  # 15 change to load_input_frames
        pipeline=train_pipeline))

# reds15/25/50 data loading
reds15_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{:08}.png'),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(
        type='RandomNoise',
        params=dict(
            noise_type=['gaussian'],
            noise_prob=[1],
            gaussian_sigma=[15, 15],  # this is a range
            gaussian_gray_noise_prob=0),  # 0 means grayscale noise is not used
        keys=['img'],
    ),
    dict(type='PackInputs')
]
reds25_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{:08}.png'),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(
        type='RandomNoise',
        params=dict(
            noise_type=['gaussian'],
            noise_prob=[1],
            gaussian_sigma=[25, 25],  # this is a range
            gaussian_gray_noise_prob=0),  # 0 means grayscale noise is not used
        keys=['img'],
    ),
    dict(type='PackInputs')
]
reds50_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{:08}.png'),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='RandomNoise',
         params=dict(
             noise_type=['gaussian'],
             noise_prob=[1],
             gaussian_sigma=[50, 50],  # this is a range
             gaussian_gray_noise_prob=0),  # 0 means grayscale noise is not used
         keys=['img'],
         ),
    dict(type='PackInputs')
]
# reds15/25/50 validation sets
reds15_val_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='reds', task_name='vsr'),
        data_root=f'{data_root}/REDS',
        data_prefix=dict(img='val_sharp_bicubic_sub', gt='val_sharp_sub'),
        num_input_frames=load_input_frames,  # add load_input_frames
        pipeline=reds15_pipeline))
reds25_val_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='reds', task_name='vsr'),
        data_root=f'{data_root}/REDS',
        data_prefix=dict(img='val_sharp_bicubic_sub', gt='val_sharp_sub'),
        num_input_frames=load_input_frames,  # add load_input_frames
        pipeline=reds25_pipeline))
reds50_val_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='reds', task_name='vsr'),
        data_root=f'{data_root}/REDS',
        data_prefix=dict(img='val_sharp_bicubic_sub', gt='val_sharp_sub'),
        num_input_frames=load_input_frames,  # add load_input_frames
        pipeline=reds50_pipeline))
# reds15/25/50 test sets
reds15_test_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='reds', task_name='vsr'),
        data_root=f'{data_root}/REDS',
        data_prefix=dict(img='val_sharp_bicubic', gt='val_sharp'),
        num_input_frames=load_input_frames,  # add load_input_frames
        pipeline=reds15_pipeline))
reds25_test_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='reds', task_name='vsr'),
        data_root=f'{data_root}/REDS',
        data_prefix=dict(img='val_sharp_bicubic', gt='val_sharp'),
        num_input_frames=load_input_frames,  # add load_input_frames
        pipeline=reds25_pipeline))
reds50_test_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='reds', task_name='vsr'),
        data_root=f'{data_root}/REDS',
        data_prefix=dict(img='val_sharp_bicubic', gt='val_sharp'),
        num_input_frames=load_input_frames,  # add load_input_frames
        pipeline=reds50_pipeline))
reds15_test_evaluator = dict(
    type='Evaluator',
    metrics=[dict(type='PSNR', convert_to='Y', prefix='reds15'),
             dict(type='SSIM', convert_to='Y', prefix='reds15'),
             dict(type='NIQE', input_order='CHW', convert_to='Y', prefix='reds15')])
reds25_test_evaluator = dict(
    type='Evaluator',
    metrics=[dict(type='PSNR', convert_to='Y', prefix='reds25'),
             dict(type='SSIM', convert_to='Y', prefix='reds25'),
             dict(type='NIQE', input_order='CHW', convert_to='Y', prefix='reds25')])
reds50_test_evaluator = dict(
    type='Evaluator',
    metrics=[dict(type='PSNR', convert_to='Y', prefix='reds50'),
             dict(type='SSIM', convert_to='Y', prefix='reds50'),
             dict(type='NIQE', input_order='CHW', convert_to='Y', prefix='reds50')])

# udm10 validation set
udm10_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{:04d}.png'),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackInputs')
]
udm10_BIx4_val_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='udm10', task_name='vsr'),
        data_root=f'{data_root}/UDM10',
        data_prefix=dict(img='BIx4_sub', gt='GT_sub'),
        num_input_frames=load_input_frames,  # add load_input_frames
        pipeline=udm10_pipeline))
# udm10 test set
udm10_BIx4_test_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='udm10', task_name='vsr'),
        data_root=f'{data_root}/UDM10',
        data_prefix=dict(img='BIx4', gt='GT'),
        num_input_frames=load_input_frames,  # add load_input_frames
        pipeline=udm10_pipeline))
udm10_BIx4_evaluator = dict(
    type='Evaluator',
    metrics=[dict(type='PSNR', convert_to='Y', prefix='UDM10-BIx4'),
             dict(type='SSIM', convert_to='Y', prefix='UDM10-BIx4'),
             dict(type='NIQE', input_order='CHW', convert_to='Y', prefix='UDM10-BIx4')])

# NTURain validation and test sets
NTURain_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{:05d}.jpg'),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackInputs')
]
NTURain_val_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='NTURain', task_name='vsr'),
        data_root=f'{data_root}/NTURain/Val',
        data_prefix=dict(img='BIx4', gt='GT'),
        num_input_frames=load_input_frames,  # add load_input_frames
        pipeline=NTURain_pipeline))
NTURain_test_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='NTURain', task_name='vsr'),
        data_root=f'{data_root}/NTURain/Test',
        data_prefix=dict(img='BIx4', gt='GT'),
        num_input_frames=load_input_frames,  # add load_input_frames
        pipeline=NTURain_pipeline))
NTURain_evaluator = dict(
    type='Evaluator', metrics=[
        dict(type='PSNR', convert_to='Y', prefix='NTURain'),
        dict(type='SSIM', convert_to='Y', prefix='NTURain'),
        dict(type='NIQE', input_order='CHW', convert_to='Y', prefix='NTURain')  # add the NIQE metric
    ])

# REVIDE validation and test sets
REVIDE_val_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{:05d}.png'),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackInputs')
]
REVIDE_val_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='REVIDE', task_name='vsr'),
        data_root=f'{data_root}/REVIDE_indoor/Val',
        data_prefix=dict(img='BIx4_sub', gt='GT_sub'),
        num_input_frames=load_input_frames,  # add load_input_frames
        pipeline=REVIDE_val_pipeline))
REVIDE_test_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{:05d}.JPG'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='PackInputs')]
REVIDE_test_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='REVIDE', task_name='vsr'),
        data_root=f'{data_root}/REVIDE_indoor/Test',
        data_prefix=dict(img='BIx4', gt='GT'),
        num_input_frames=load_input_frames,  # add load_input_frames
        pipeline=REVIDE_test_pipeline))
REVIDE_evaluator = dict(
    type='Evaluator', metrics=[
        dict(type='PSNR', convert_to='Y', prefix='REVIDE'),
        dict(type='SSIM', convert_to='Y', prefix='REVIDE'),
        dict(type='NIQE', input_order='CHW', convert_to='Y', prefix='REVIDE')  # add the NIQE metric
    ])

# video_lq test set
video_lq_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{:08d}.png'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='PackInputs')
]
video_lq_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='video_lq', task_name='vsr'),
        data_root=f'{data_root}/VideoLQ',
        data_prefix=dict(img='', gt=''),
        num_input_frames=load_input_frames,  # add load_input_frames
        pipeline=video_lq_pipeline))
video_lq_evaluator = dict(
    type='Evaluator',
    metrics=[dict(type='NIQE', input_order='CHW', convert_to='Y', prefix='Video_LQ')])

# multiple validation
val_pipeline = [
    reds15_pipeline,
    reds25_pipeline,
    reds50_pipeline,
    udm10_pipeline,
    NTURain_pipeline,
    REVIDE_val_pipeline,
]
val_dataloader = [
    reds15_val_dataloader,
    reds25_val_dataloader,
    reds50_val_dataloader,
    udm10_BIx4_val_dataloader,
    NTURain_val_dataloader,
    REVIDE_val_dataloader,
]
val_evaluator = [
    reds15_test_evaluator,
    reds25_test_evaluator,
    reds50_test_evaluator,
    udm10_BIx4_evaluator,
    NTURain_evaluator,
    REVIDE_evaluator,
]

# multiple testing
test_pipeline = [
    reds15_pipeline,
    reds25_pipeline,
    reds50_pipeline,
    udm10_pipeline,
    NTURain_pipeline,
    REVIDE_test_pipeline,
    video_lq_pipeline
]
test_dataloader = [
    reds15_test_dataloader,
    reds25_test_dataloader,
    reds50_test_dataloader,
    udm10_BIx4_test_dataloader,
    NTURain_test_dataloader,
    REVIDE_test_dataloader,
    video_lq_dataloader
]
test_evaluator = [
    reds15_test_evaluator,
    reds25_test_evaluator,
    reds50_test_evaluator,
    udm10_BIx4_evaluator,
    NTURain_evaluator,
    REVIDE_evaluator,
    video_lq_evaluator
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=300_000, val_interval=2000)
val_cfg = dict(type='MultiValLoop')
test_cfg = dict(type='MultiTestLoop')

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    accumulative_counts=5,  # gradient accumulation
    optimizer=dict(type='Adam', lr=2e-4, betas=(0.9, 0.99)),
    paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.125)}),
    clip_grad=dict(max_norm=1, norm_type=2),  # norm-based gradient clipping
)

# compile the model to accelerate training
cfg = dict(compile='compile_options')

# learning policy
param_scheduler = dict(
    type='CosineRestartLR',
    by_epoch=False,
    periods=[300000],
    restart_weights=[1],
    eta_min=1e-7)


default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=2000,
        save_optimizer=True,
        out_dir=save_dir,
        max_keep_ckpts=20,
        # save_best='PSNR',
        # rule='greater',
        save_best=['reds15/PSNR', 'reds25/PSNR', 'reds50/PSNR', 'UDM10-BIx4/NIQE', 'NTURain/PSNR', 'REVIDE/PSNR'],
        rule=['greater', 'greater', 'greater', 'less', 'greater', 'greater'],
        by_epoch=False),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

# image visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ConcatImageVisualizer',
    vis_backends=vis_backends,
    fn_key='gt_path',
    img_keys=['input', 'pred_img', 'gt_img'],  # 'input'->'img'
    bgr2rgb=True)

# custom hook
custom_hooks = [
    dict(type='BasicVisualizationHook', interval=5),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema'),
        interval=1,
        interp_cfg=dict(momentum=0.001),
    )
]

model_wrapper_cfg = dict(
    type='MMSeparateDistributedDataParallel',
    broadcast_buffers=False,
    find_unused_parameters=False)    # True->False
