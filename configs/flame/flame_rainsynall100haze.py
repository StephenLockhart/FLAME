# FlameNet hybrid model training config - RainSynAll100Haze dataset, 256-channel / 160-patch version (64A+32 dual-point enhancement)
# optical flow: simplified single-interval pa_frames=2 + random adaptive flow_mask
# 64a32: 64A+32 dual-point enhancement strategy (trsa64a=True, trsa32=True, trsa64b=False)
# f256v48: Mamba feature dimension 256, VRT processing dimension 48
# g8dp2: deformable_groups=8, depth=2 (two-layer TRSA processing)
# p160: 160×160 patch size
# RainSynAll100Haze: train on the RainSynAll100_Haze dataset
# lr1e-4: learning rate 1e-4
# 300k: 300K training iterations (referencing the proven VRDS config)
# 🔧 FLOW_FIXED: the complex optical-flow accumulation issue has been fixed by simplifying to single-interval pa_frames=2
# 📦 Dependencies: a standard MMagic installation is sufficient
default_scope = 'mmagic'  # default registry scope for module lookup; otherwise mmengine raises errors

load_batch_size = 4
load_input_frames = 6  # FlameNet uses 6 frames, aligned with VRT
input_val_frames = 7  # 🔧 use all 7 frames for validation
input_frames = load_input_frames  # use the original frame count directly
iter_k = 300  # 🔧 300K training iterations, referencing the proven VRDS config
iters = 1000 * iter_k
interval_val = 10000  # 🔧 fix: increase the validation interval to reduce validation frequency and avoid dataset issues

experiment_name = 'flame_rainsynall100haze'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs'

scale = 1   # video restoration mode: deraining and dehazing

load_from = None
resume = False


# model settings
model = dict(
    type='FlameDynamic',  # 🔧 use the dynamic Hilbert pipeline, supports 128 patch
    generator=dict(
        type='FlameNet160',  # 🎯 dedicated version for 160 patch: resolves hard-coding issues, supports flows_40/flows_20
        num_features=256,                    # 🔧 256 channels, referencing the proven VRDS config
        vrt_dim=48,                         # 48 → 48, more consistent with VRT
        scale_factor=scale,
        img_size=[6, 40, 40],               # 🔧 key change: 160 patch corresponds to a 40×40 feature map (160/4=40)
        window_size=[6, 8, 8],              # temporal × spatial window
        pa_frames=2,                        # 6 → 2, single-interval optical flow only, avoiding accumulated error
        feat_pretrained='https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth',
        spynet_path='https://github.com/JingyunLiang/VRT/releases/download/v0.0/spynet_sintel_final-3d2a1287.pth',
        
        # 🔥 FLAME-specific configuration (core fixes)
        use_flow_mask=True,              # flow mask switch; enables adaptive adjustment
        flow_mask_strength=0.5,          # baseline value of the adaptive strength
        
        # 🎯 TRSA switch configuration (64A+32 dual-point enhancement strategy)
        trsa64a=True,   # enable 64A (high-resolution detail enhancement)
        trsa32=True,    # enable 32 (mid-resolution enhancement)
        trsa64b=False,  # disable 64B (simplify the architecture to avoid complexity)
        
        # 🔧 TRSA parameter configuration (following the VRT design)
        deformable_groups=8,            # 48/8=6, following the VRT video deblurring config  
        trsa64_heads=6,                  # 48/6=8, consistent with VRT
        trsa64_depth=2,                  # keep a depth of 2 layers
        trsa32_heads=6,                  # consistent with VRT
        trsa32_depth=2,                  # keep a depth of 2 layers

        
        # 🔧 DDP compatibility configuration (fixes checkpoint conflicts)
        use_checkpoint_attn=False,       # 🔧 DDP fix: forcibly disable attention checkpoint
        use_checkpoint_ffn=False,        # 🔧 DDP fix: forcibly disable ffn checkpoint
        
        # fixed config: frequency enhancement enabled, no contrastive learning
        fre_decoder=True,   # enable frequency enhancement
    ),
    num_input_frames=load_input_frames,
    scale_factor=scale,
    train_cfg=dict(),
    test_cfg=dict(
        scale_factor=scale,
        window_size=[2, int(8 / scale), int(8 / scale)],  # temporal and spatial window size
        tile=[
            load_input_frames,  # if 0, feed the full-length video
            int(160 / scale),   # 🔧 key change: 160×160 inference patch
            int(160 / scale),
        ],  # [temporal segment size, spatial patch height, spatial patch width]
        tile_overlap=[
            load_input_frames - 1,
            int(120 / scale),    # 🔧 corresponding overlap: 120×120 (160*0.75)
            int(120 / scale),
        ],  # [temporal overlap, spatial height overlap, spatial width overlap]
        use_temporal_gradient=True,  # whether to use Gaussian weights along time
        use_temporal_average=False,  # whether to use cumulative averaging along time
        use_spatial_gradient=True,   # whether to use Gaussian weights spatially
        use_spatial_average=False,   # whether to use cumulative averaging spatially
    ),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),     
    # consistent with the derainer: use vgg16
    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={
            '3': 1.0,    # relu1_2
            '8': 1.0,    # relu2_2
            '15': 0.5,   # relu3_3
        },
        vgg_type='vgg16',
        criterion='mse',
        perceptual_weight=0.02,  # perceptual loss weight, kept unchanged
        style_weight=0,
        norm_img=False,
        pretrained='torchvision://vgg16'  
    ),
    # fixed version: frequency enhancement but no contrastive learning
    ensemble=dict(type='SpatialTemporalEnsemble', is_temporal_ensemble=False),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
    ),
)

train_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1], filename_tmpl='{:d}.jpg', start_idx=1),  # 🔧 RainSynAll100: numeric jpg format starting from 1
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(type='PairedRandomCrop', gt_patch_size=160),  # 🔧 160 patch config, fully utilizing the dataset's minimum size of 165
    dict(type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['img', 'gt'], transpose_ratio=0.5),

    dict(
        type='ColorJitter',   # add pixel-level data augmentation
        keys=['img', 'gt'],
        channel_order='rgb',
        brightness=0.05,
        contrast=0.05,
        saturation=0.05,
        hue=0.05),
    dict(type='Clip', keys=['img']),  # clamp pixel values to 0-1, preventing degraded img from going out of range
    dict(type='PackInputs'),
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='PackInputs'),
]

# 🔧 RainSynAll100Haze dataset configuration
data_root = 'data/RainSynAll100'  # TODO: replace with your local RainSynAll100 dataset root

train_dataloader = dict(
    num_workers=10,
    batch_size=load_batch_size,
    drop_last=True,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='RainSynAll100Haze_Train', task_name='vr'),
        data_root=f'{data_root}/Video_rain_synthesis_train',
        data_prefix=dict(img='Rain_Haze', gt='GT'),  # 🔧 RainSynAll100Haze path structure
        depth=1,
        num_input_frames=load_input_frames,
        pipeline=train_pipeline,
    ),
)

val_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{:d}.jpg',  # 🔧 RainSynAll100: original jpg format
        start_idx=1,  # 🔧 numbering starts from 1
    ),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackInputs'),
]

# 🔧 RainSynAll100Haze test set configuration
data_test = f'{data_root}/Video_rain_synthesis_test'

RainSynAll100Haze_val_dataloader = dict(
    num_workers=10,
    batch_size=1,  # 🔧 fix: validation batch_size=1, avoiding stack conflicts from videos of different sizes
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='RainSynAll100Haze_Test', task_name='vr'),
        data_root=data_test,
        data_prefix=dict(img='Rain_Haze', gt='GT'),  # 🔧 RainSynAll100Haze test set path structure
        num_input_frames=input_val_frames,
        pipeline=val_pipeline,
    ),
)

RainSynAll100Haze_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(type='PSNR', convert_to='Y', prefix='RainSynAll100Haze'),
        dict(type='SSIM', convert_to='Y', prefix='RainSynAll100Haze'),
    ],
)

# validation and test configuration (single dataset)
val_dataloader = [RainSynAll100Haze_val_dataloader]
val_evaluator = [RainSynAll100Haze_evaluator]

test_dataloader = val_dataloader
test_evaluator = val_evaluator

train_cfg = dict(type='IterBasedTrainLoop', max_iters=iters, val_interval=interval_val)
val_cfg = dict(type='MultiValLoop')
test_cfg = dict(type='MultiTestLoop')

# optimizer - FlameNet 256-channel config (fine-grained parameter grouping strategy)
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',  # 🔧 use a standard optimizer wrapper, stable and reliable
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0001),
    # 🔥 256-channel version: dedicated parameter-grouping optimization strategy for FlameNet
    paramwise_cfg=dict(
        custom_keys={
            'spynet': dict(lr_mult=0.25),        # SpyNet uses 1/4 learning rate (following the standard config)
            'vrt_stage': dict(lr_mult=1.5),      # VRT Stage uses a larger learning rate
            'advanced_fusion': dict(lr_mult=2), # larger learning rate for the other parameters of the advanced fusion layer
        }
    ),
    clip_grad=dict(max_norm=4.0, norm_type=2),  # 🔧 FlameNet: relatively strong gradient clipping
)

# model compilation to speed up training
cfg = dict(compile='compile_options')

# learning policy - FlameNet 256-channel learning rate schedule (300K total training: 50K warm-up + 150K stable + 100K decay)
param_scheduler = [
    # 🔧 FlameNet 256-channel warmup: 50k warm-up period to stabilize the complex optical-flow fusion
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=50000,  # 50k warmup, letting FlowGuidedFusion activate gradually
    ),
    # 🔧 FlameNet 256-channel main training: cosine annealing (starts at 200k, corresponding to 300k total training)
    dict(
        type='CosineRestartLR',
        periods=[(iters-200000)],  # remaining 100k iterations (300k-200k=100k)
        restart_weights=[1],
        by_epoch=False,
        begin=200000,
        end=iters,
        eta_min=1e-7
    )
]


default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=interval_val,  # validation interval
        save_optimizer=True,
        out_dir=save_dir,
        max_keep_ckpts=5,  # keep more checkpoints, since multiple best metrics are saved
        save_best=[
            'RainSynAll100Haze/PSNR',    # Peak Signal-to-Noise Ratio (primary metric)
            'RainSynAll100Haze/SSIM',    # Structural Similarity Index
        ],
        rule=['greater', 'greater'],  # higher is better for both PSNR and SSIM
        by_epoch=False,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(
        mp_start_method='fork', opencv_num_threads=4
    ),
    dist_cfg=dict(backend='nccl'),
)

log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=100, by_epoch=False)


# visualizer
vis_backends = [
    dict(type='LocalVisBackend'),
    # Optional: enable Weights & Biases logging by uncommenting below and
    # setting your own project name (requires `pip install wandb` and login).
    # dict(
    #     type='WandbVisBackend',
    #     init_kwargs=dict(
    #         project='FLAME',
    #         name='flame_rainsynall100haze'
    #     ),
    #     save_dir=work_dir
    # )
]

visualizer = dict(
    type='ConcatImageVisualizer',
    vis_backends=vis_backends,
    fn_key='gt_path',
    img_keys=['input', 'pred_img', 'gt_img'],
    bgr2rgb=True,
)

# visualization hooks
custom_hooks = [
    dict(type='BasicVisualizationHook', interval=5),  # 🔧 fix: reduce the interval, following BasicVSR
    dict(
        type='ExponentialMovingAverageHook',  # 🔧 fix: add the EMA hook to improve visualization
        module_keys=('generator_ema'),
        interval=1,
        interp_cfg=dict(momentum=0.001),
    )
]

model_wrapper_cfg = dict(
    type='MMSeparateDistributedDataParallel',
    broadcast_buffers=False,
    find_unused_parameters=True,  # set to True to avoid FLAME parameter issues
)

# FlameNet 256-channel / 160-patch RainSynAll100Haze dataset configuration notes
"""
🎯 FlameNet 256-channel / p160 version (optical-flow fix + adaptive mask) - RainSynAll100Haze dataset configuration notes (160 patch + dynamic Hilbert pipeline)

📊 Naming convention:
- f: frequency enhancement (fre_decoder=True)
- 64a32: 64A+32 dual-point enhancement strategy (trsa64a=True, trsa32=True, trsa64b=False)
- f256v48: Mamba feature dimension 256, VRT processing dimension 48
- g8dp2: deformable_groups=8, depth=2 (two-layer TRSA processing depth)
- p160: 160×160 inference patch size
- RainSynAll100Haze: train on the RainSynAll100_Haze dataset
- lr1e-4: learning rate 1e-4
- 300k: 300K training iterations (referencing the proven VRDS config)

🔧 Model configuration:
1. **Wrapper**: FlameDynamic (dynamic Hilbert pipeline, supports arbitrary patch size)
2. **Network type**: FlameNet160 (dedicated version for 160 patch, 256 channels)
3. **Training patch**: 160×160, i.e. img_size=[6,40,40] (160/4) feature maps
4. **Inference patch**: tile=[6,160,160], tile_overlap=[5,120,120]
5. **VRT config**: vrt_dim=48, deformable_groups=8, trsa64a/trsa32 enabled, pa_frames=2

📁 RainSynAll100Haze dataset structure:
- Training set:
  - GT: data/RainSynAll100/Video_rain_synthesis_train/GT/0001/1.jpg
  - IMG: data/RainSynAll100/Video_rain_synthesis_train/Rain_Haze/0001/1.jpg
- Test set:
  - GT: data/RainSynAll100/Video_rain_synthesis_test/GT/0900/1.jpg  
  - IMG: data/RainSynAll100/Video_rain_synthesis_test/Rain_Haze/0900/1.jpg

📊 Dataset characteristics:
- 999 training sequences; the last 100 overlap with the test set
- Rain-haze mixture degradation, more challenging than rain alone
- Frame sequences are numbered from 1, suitable for pursuing SOTA metrics
- GT is shared with RainSynAll100; IMG is the Rain_Haze version

💡 Core advantages of 256 channels:
1. 🎯 Referencing the proven VRDS config: uses a validated, stable parameter combination
2. 🔧 Numerical stability: 256 channels deliver better stability across multiple datasets
3. ⚡ 160×160 patch: preserves detail restoration capability while fitting the dataset's image sizes
4. 🛡️ Stable dual-point fusion: 64A+32 co-processing runs more smoothly
5. 📈 Visualization fix: an EMA hook is added to improve image outputs

🎯 Current configuration characteristics:
- Dataset: RainSynAll100_Haze, a rain-haze mixture degradation video dataset
- File format: JPG files with numeric names starting from 1 ({:d}.jpg)
- Patch size: 160×160, corresponding to a 40×40 feature map
- Inference config: tile=[6,160,160], tile_overlap=[5,120,120]
- Validation interval: 10000
- Training iterations: 300K (50K warm-up + 150K stable + 100K decay)
- Dual-point enhancement: trsa64a=True, trsa32=True, trsa64b=False
- Parameter optimization: the VRT module uses a larger learning rate (1.5x-2x)
- Single-dataset training: focused on RainSynAll100_Haze performance optimization

📊 Evaluation metric configuration:
- PSNR: Peak Signal-to-Noise Ratio (primary metric)
- SSIM: Structural Similarity Index

📁 Data format notes:
- Filename format: 1.jpg, 2.jpg, 3.jpg... (single-digit number, starting from 1)
- filename_tmpl='{:d}.jpg' together with start_idx=1 correctly supports this format
- Different from the common {:05d}.jpg (00001.jpg) format

💾 Best-model saving strategy:
- Automatically saves the best models for the two metrics PSNR and SSIM
- PSNR↑, SSIM↑ (higher is better)
- Keeps at most 5 checkpoint files

🚀 Training command:
python tools/train.py configs/flame/flame_rainsynall100haze.py

Expected: the 256-channel / 160-patch version achieves stable training results on RainSynAll100_Haze; rain-haze mixture degradation is highly challenging
""" 
