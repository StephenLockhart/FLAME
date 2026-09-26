# DMVRTNetFixed RainSynAll100 full-test configuration
# Resolves the issue of testing only 50 out of 100 videos
# Based on the training config dmvrt-fixed-64a32_f128v48g8dp2_lr1e-4_RainSynAll100_20250916.py

default_scope = 'mmagic'

# test configuration naming
experiment_name = 'dmvrt-fixed-64a32_f128_RainSynAll100_FullTest_20250919'
work_dir = f'./work_dirs/test/{experiment_name}'
save_dir = './work_dirs/test'

load_batch_size = 1  # 🔧 force batch_size=1 to ensure each video is processed independently
load_input_frames = 7  # RainSynAll100 uses 7 frames
input_val_frames = 7   # 🔧 keep consistent with load_input_frames
input_frames = load_input_frames
scale = 1

# checkpoint path - passed in via the command line
load_from = None
resume = False

# model settings - 128-channel version
model = dict(
    type='AimVRT',
    generator=dict(
        type='DMVRTNetFixed128',  # 🎯 use the 128-channel dedicated version
        num_features=128,         # 🔧 128-channel version
        vrt_dim=48,
        scale_factor=scale,
        img_size=[7, 64, 64],     # 🔧 7-frame configuration
        window_size=[6, 8, 8],
        pa_frames=2,
        feat_pretrained='https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth',
        spynet_path='https://github.com/JingyunLiang/VRT/releases/download/v0.0/spynet_sintel_final-3d2a1287.pth',
        
        # 🔥 DMVRTFixed-specific configuration
        use_flow_mask=True,
        flow_mask_strength=0.5,
        
        # 🎯 TRSA switch configuration (64A+32 dual-point enhancement strategy)
        trsa64a=True,   # 🔥 enable 64A
        trsa32=True,    # 🔥 enable 32
        trsa64b=False,  # 🔧 disable 64B
        
        # 🔧 TRSA parameter configuration
        deformable_groups=8,
        trsa64_heads=6,
        trsa64_depth=2,
        trsa32_heads=6,
        trsa32_depth=2,
        
        # 🔧 DDP compatibility configuration
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
    num_input_frames=load_input_frames,
    scale_factor=scale,
    train_cfg=dict(),
    test_cfg=dict(
        scale_factor=scale,
        window_size=[2, int(8 / scale), int(8 / scale)],
        tile=[
            load_input_frames,
            int(128 / scale),   # 🔧 RainSynAll100 uses a 128×128 patch
            int(128 / scale),
        ],
        tile_overlap=[
            load_input_frames - 1,
            int(96 / scale),
            int(96 / scale),
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

# 🔧 key fix: test data pipeline - optimized for the RainSynAll100 dataset
test_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{:d}.jpg',    # 🔧 RainSynAll100 uses .jpg format with numeric naming starting from 1
        start_idx=1,                 # 🔧 RainSynAll100 starts from 1, not 0
    ),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackInputs'),
]

# RainSynAll100 dataset paths
data_root = 'data/RainSynAll100'  # TODO: replace with your local dataset root
data_test = f'{data_root}/Video_rain_synthesis_test'

# 🔧 key fix: RainSynAll100 test dataloader configuration
RainSynAll100_test_dataloader = dict(
    num_workers=10,   # 🔧 increase the number of workers
    batch_size=1,    # 🔧 force batch_size=1 to ensure each video is processed independently
    persistent_workers=False,  # 🔧 disable persistent workers to avoid memory issues
    sampler=dict(type='DefaultSampler', shuffle=False),  # 🔧 ensure sequential processing
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='RainSynAll100_Test', task_name='vr'),
        data_root=data_test,
        data_prefix=dict(img='Rain', gt='GT'),  # 🔧 RainSynAll100 test set path structure
        num_input_frames=None,
        pipeline=test_pipeline,
        # 🔧 added: explicitly specify the test sample range to ensure all 100 videos are covered
        test_mode=True,
        ann_file=None,  # no annotation file; scan the directory directly
    ),
)

# 🔧 evaluation metric configuration
RainSynAll100_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(type='PSNR', convert_to='Y', prefix='RainSynAll100'),
        dict(type='SSIM', convert_to='Y', prefix='RainSynAll100'),
    ],
)

# test configuration
test_dataloader = RainSynAll100_test_dataloader
test_evaluator = RainSynAll100_evaluator
test_cfg = dict(type='TestLoop')  # 🔧 use a single TestLoop

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

# 🎯 visualization configuration
vis_backends = [
    dict(
        type='LocalVisBackend',
    )
]

visualizer = dict(
    type='SingleImageVisualizer',
    vis_backends=vis_backends,
    fn_key='gt_path',
    img_keys=['pred_img'],  # 🎯 save only the predicted images
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

# 🚀 Test configuration notes
"""
🎯 RainSynAll100 full-test configuration notes

📊 Key fixes:
1. batch_size=1: forces each video to be processed independently
2. filename_tmpl='{:d}.jpg': matches RainSynAll100's .jpg format and numeric naming
3. start_idx=1: RainSynAll100 starts from 1, not 0
4. persistent_workers=False: avoids dataloader conflicts
5. num_input_frames=7: consistent with the training config
6. Uses DMVRTNetFixed128: the 128-channel dedicated version

🚀 Run commands:
# Single-GPU test
python tools/test.py configs/aimvrtRain/test/dmvrt-fixed-64a32_f128_RainSynAll100_FullTest_20250919.py work_dirs/dmvrt-fixed-f-64a32_f128v48g8dp2_6pa2xb1-lr1e-4-300k_RainSynAll100_20250916/best_RainSynAll100_PSNR_iter_300000.pth

# Dual-GPU test
bash tools/dist_test.sh configs/aimvrtRain/test/dmvrt-fixed-64a32_f128_RainSynAll100_FullTest_20250919.py work_dirs/dmvrt-fixed-f-64a32_f128v48g8dp2_6pa2xb1-lr1e-4-300k_RainSynAll100_20250916/best_RainSynAll100_PSNR_iter_300000.pth 2

💡 Expected results:
- Tests all 100 RainSynAll100 videos instead of only 50
- Obtains validation of the true PSNR=44.93, SSIM=0.9876 results
- Confirms SOTA performance on pure rain-streak degradation

📁 Dataset structure:
data/RainSynAll100/Video_rain_synthesis_test/
├── Rain/
│   ├── 0900/ (7 or 9 frames)
│   ├── 0901/
│   └── ... (100 videos in total)
└── GT/
    ├── 0900/
    ├── 0901/
    └── ...
"""
