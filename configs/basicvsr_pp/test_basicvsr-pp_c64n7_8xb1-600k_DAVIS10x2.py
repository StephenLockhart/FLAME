# _base_ = '../basicvsr/basicvsr_2xb4_reds4.py'
_base_ = '../_base_/default_runtime.py'

experiment_name = 'test_basicvsr-pp_c64n7_8xb1-600k_DAVIS10x2'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs'

# model settings
model = dict(
    type='BasicVSRx2',
    generator=dict(
        type='BasicVSRPlusPlusNet',
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=True,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth'),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    train_cfg=dict(fix_iter=5000),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackInputs')
]

# train_dataloader = dict(
#     num_workers=6, batch_size=1, dataset=dict(num_input_frames=30))

# config for vid4
vid4_data_root = 'data/Vid4x2'

vid4_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackInputs')
]
vid4_bd_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    # dict(
    #     type='RandomBlur',
    #     params=dict(
    #         kernel_size=[11],
    #         kernel_list=['iso'],
    #         kernel_prob=[1],
    #         sigma_x=[1.6, 1.6],
    #         # sigma_y=[0, 1.6],
    #     ),
    #     keys=['img'],),
    # dict(
    #     type='Resize',
    #     scale=1 / 4,
    #     keep_ratio=True,
    #     interpolation='bicubic',
    #     backend='pillow',
    #     keys=['img'],),
    dict(type='PackInputs')
]


vid4_bd_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='vid4', task_name='vsr'),
        data_root=vid4_data_root,
        data_prefix=dict(img='BDx2', gt='GT'),
        ann_file='meta_info_Vid4_GT.txt',
        depth=1,
        # num_input_frames=30,
        # fixed_seq_len=30,
        pipeline=vid4_bd_pipeline))

vid4_bi_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='vid4', task_name='vsr'),
        data_root=vid4_data_root,
        data_prefix=dict(img='BIx2', gt='GT'),
        ann_file='meta_info_Vid4_GT.txt',
        depth=1,
        # num_input_frames=30,
        # fixed_seq_len=30,
        pipeline=vid4_pipeline))

vid4_bd_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='VID4-BDx2'),
    dict(type='SSIM', convert_to='Y', prefix='VID4-BDx2'),
]
vid4_bi_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='VID4-BIx2'),
    dict(type='SSIM', convert_to='Y', prefix='VID4-BIx2'),
]

# config for DAIVS-10
DAVIS10_data_root = 'data/DAVIS-10x2'

DAVIS10_pipeline = [
    # dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='GenerateSegmentIndices', interval_list=[1], filename_tmpl='{:05d}.jpg'),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackInputs')
]

DAVIS10_bd_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='DAVIS10', task_name='vsr'),
        data_root=DAVIS10_data_root,
        data_prefix=dict(img='BDx2', gt='GT'),
        depth=1,
        num_input_frames=20,
        # fixed_seq_len=20,
        pipeline=DAVIS10_pipeline))

DAVIS10_bi_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='DAVIS10', task_name='vsr'),
        data_root=DAVIS10_data_root,
        data_prefix=dict(img='BIx2', gt='GT'),
        depth=1,
        num_input_frames=20,
        # fixed_seq_len=20,
        pipeline=DAVIS10_pipeline))

DAVIS10_bd_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='DAVIS10-BDx2'),
    dict(type='SSIM', convert_to='Y', prefix='DAVIS10-BDx2'),
]
DAVIS10_bi_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='DAVIS10-BIx2'),
    dict(type='SSIM', convert_to='Y', prefix='DAVIS10-BIx2'),
]

# config for test
test_cfg = dict(type='MultiTestLoop')
test_dataloader = [
    # vid4_bd_dataloader,
    # vid4_bi_dataloader,
    # DAVIS10_bd_dataloader,
    DAVIS10_bi_dataloader,
]
test_evaluator = [
    # vid4_bd_evaluator,
    # vid4_bi_evaluator,
    # DAVIS10_bd_evaluator,
    DAVIS10_bi_evaluator,
]

# train_cfg = dict(
#     type='IterBasedTrainLoop', max_iters=600_000, val_interval=5000)

# # optimizer
# optim_wrapper = dict(
#     constructor='DefaultOptimWrapperConstructor',
#     type='OptimWrapper',
#     optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99)),
#     paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.25)}))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ConcatImageVisualizer',
    vis_backends=vis_backends,
    fn_key='gt_path',
    img_keys=['input', 'pred_img', 'gt_img'],
    bgr2rgb=True)

default_hooks = dict(checkpoint=dict(out_dir=save_dir))

# # learning policy
# param_scheduler = dict(
#     type='CosineRestartLR',
#     by_epoch=False,
#     periods=[600000],
#     restart_weights=[1],
#     eta_min=1e-7)
