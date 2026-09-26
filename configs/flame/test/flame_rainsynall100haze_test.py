# FlameNet160 RainSynAll100Haze full-test configuration
#
# Test config for the released 256-channel / 160-patch FLAME model
# (FlameDynamic + FlameNet160). Model parameters follow the main
# training config flame_rainsynall100haze.py
# and the mmengine backup config recorded in the training work_dir
# (work_dirs/...RainSynAll100Haze_20250922/20250924_104310/vis_data/config.py),
# which is also the setup that produced the reported validation PSNR/SSIM.
#
# Frame setup: each RainSynAll100Haze test video contains 7 frames, while the
# model is trained on 6-frame clips (img_size=[6, 40, 40]). The 7 input
# frames are therefore tiled into overlapping 6-frame clips with
# tile=[6, 160, 160] and tile_overlap=[5, 120, 120].

default_scope = 'mmagic'

# test configuration naming
experiment_name = 'flame_rainsynall100haze_test'
work_dir = f'./work_dirs/test/{experiment_name}'
save_dir = './work_dirs/test'

load_batch_size =  1  # force batch_size=1 so each video is processed independently
num_clip_frames = 6   # trained clip length (model img_size temporal dim)
load_input_frames = 7  # RainSynAll100Haze test videos have 7 frames
input_val_frames = 7   # keep consistent with load_input_frames
input_frames = num_clip_frames
scale = 1

# checkpoint path - passed in via the command line
load_from = None
resume = False

# model settings - 256-channel / 160-patch version (FlameDynamic wrapper)
model = dict(
    type='FlameDynamic',
    generator=dict(
        type='FlameNet160',  # dedicated version for 160x160 patches
        num_features=256,         # 256-channel version
        vrt_dim=48,
        scale_factor=scale,
        img_size=[6, 40, 40],     # 6-frame clip, 160/4 = 40x40 feature map
        window_size=[6, 8, 8],
        pa_frames=2,
        feat_pretrained='https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth',
        spynet_path='https://github.com/JingyunLiang/VRT/releases/download/v0.0/spynet_sintel_final-3d2a1287.pth',

        # FLAME-specific configuration
        use_flow_mask=True,
        flow_mask_strength=0.5,

        # TRSA switch configuration (64A+32 dual-point enhancement strategy)
        trsa64a=True,   # enable 64A
        trsa32=True,    # enable 32
        trsa64b=False,  # disable 64B

        # TRSA parameter configuration
        deformable_groups=8,
        trsa64_heads=6,
        trsa64_depth=2,
        trsa32_heads=6,
        trsa32_depth=2,

        # DDP compatibility configuration
        use_checkpoint_attn=False,
        use_checkpoint_ffn=False,

        fre_decoder=True,
    ),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={
            '3': 1.0,
            '8': 1.0,
            '15': 0.5,
        },
        vgg_type='vgg16',
        criterion='mse',
        perceptual_weight=0.02,
        style_weight=0,
        norm_img=False,
        pretrained='torchvision://vgg16'
    ),
    ensemble=dict(type='SpatialTemporalEnsemble', is_temporal_ensemble=False),
    num_input_frames=num_clip_frames,
    scale_factor=scale,
    train_cfg=dict(),
    test_cfg=dict(
        scale_factor=scale,
        window_size=[2, int(8 / scale), int(8 / scale)],
        tile=[
            num_clip_frames,
            int(160 / scale),   # 160x160 inference patch
            int(160 / scale),
        ],
        tile_overlap=[
            num_clip_frames - 1,
            int(120 / scale),   # 120x120 overlap
            int(120 / scale),
        ],
        use_temporal_gradient=True,
        use_temporal_average=False,
        use_spatial_gradient=True,
        use_spatial_average=False,
    ),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
    ),
)

# test data pipeline - RainSynAll100Haze uses .jpg files numbered from 1
test_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{:d}.jpg',
        start_idx=1,
    ),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackInputs'),
]

# RainSynAll100 (RainSynAll100Haze shares the same dataset root)
data_root = 'data/RainSynAll100'  # TODO: replace with your local dataset root
data_test = f'{data_root}/Video_rain_synthesis_test'

# RainSynAll100Haze test dataloader configuration
RainSynAll100Haze_test_dataloader = dict(
    num_workers=10,
    batch_size=1,    # force batch_size=1 so each video is processed independently
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='RainSynAll100Haze_Test', task_name='vr'),
        data_root=data_test,
        data_prefix=dict(img='Rain_Haze', gt='GT'),
        num_input_frames=load_input_frames,
        pipeline=test_pipeline,
        test_mode=True,
        ann_file=None,  # no annotation file; scan the directory directly
    ),
)

# evaluation metric configuration
RainSynAll100Haze_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(type='PSNR', convert_to='Y', prefix='RainSynAll100Haze'),
        dict(type='SSIM', convert_to='Y', prefix='RainSynAll100Haze'),
    ],
)

# test configuration
test_dataloader = RainSynAll100Haze_test_dataloader
test_evaluator = RainSynAll100Haze_evaluator
test_cfg = dict(type='TestLoop')

# environment configuration
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(
        mp_start_method='fork',
        opencv_num_threads=4
    ),
    dist_cfg=dict(backend='nccl'),
)

log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=100, by_epoch=False)

# visualization configuration
vis_backends = [
    dict(
        type='LocalVisBackend',
    )
]

visualizer = dict(
    type='SingleImageVisualizer',
    vis_backends=vis_backends,
    fn_key='gt_path',
    img_keys=['pred_img'],  # save only the predicted images
    bgr2rgb=True,
)

# visualization hooks
custom_hooks = [
    dict(
        type='BasicVisualizationHook',
        interval=1,  # save every test sample
    )
]

# distributed configuration
model_wrapper_cfg = dict(
    type='MMSeparateDistributedDataParallel',
    broadcast_buffers=False,
    find_unused_parameters=True,
)

# Test configuration notes
"""
RainSynAll100Haze full-test configuration notes

Key points:
1. batch_size=1: each video is processed independently.
2. filename_tmpl='{:d}.jpg' with start_idx=1 matches the dataset naming.
3. 7 frames are loaded per test video; the model runs on overlapping
   6-frame clips (tile=[6, 160, 160], tile_overlap=[5, 120, 120]), exactly
   as in the training-time validation that produced the reported metrics.
4. FlameDynamic + FlameNet160, num_features=256, vrt_dim=48,
   img_size=[6, 40, 40].
5. data_prefix img='Rain_Haze', gt='GT'.

Run commands:
# Single-GPU test (best-PSNR checkpoint)
python tools/test.py configs/flame/test/flame_rainsynall100haze_test.py checkpoints/flame_rainsynall100haze_psnr.pth

# Single-GPU test (best-SSIM checkpoint)
python tools/test.py configs/flame/test/flame_rainsynall100haze_test.py checkpoints/flame_rainsynall100haze_ssim.pth

Dataset structure:
data/RainSynAll100/Video_rain_synthesis_test/
├── Rain_Haze/
│   ├── 0900/ (7 frames)
│   ├── 0901/
│   └── ... (100 videos in total)
└── GT/
    ├── 0900/
    ├── 0901/
    └── ...
"""
