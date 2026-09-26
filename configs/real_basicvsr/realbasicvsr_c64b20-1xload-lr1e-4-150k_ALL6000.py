_base_ = './realbasicvsr_wogan-c64b20-1xload-lr1e-4-300k_ALL6000.py'

load_batch_size = 3
load_input_frames = 10
name_input_frames = 2 * load_input_frames  # doubled because flip augmentation doubles the data
experiment_name = f'realbasicvsr_c64b20-1x{name_input_frames}xb{load_batch_size}-lr5e-5-150k_ALL6000'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

scale = 4

# model settings
model = dict(
    type='RealBasicVSR',
    generator=dict(
        type='RealBasicVSRNet',
        mid_channels=64,
        num_propagation_blocks=20,
        num_cleaning_blocks=20,
        dynamic_refine_thres=255,  # change to 5 for test
        spynet_pretrained='ckpt/spynet_20210409-c6c1bd09.pth',
        is_fix_cleaning=True,   # freeze the cleaning module
        is_sequential_cleaning=False),
    discriminator=dict(
        type='UNetDiscriminatorWithSpectralNorm',
        in_channels=3,
        mid_channels=64,
        skip_connection=True),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),     # per the paper, l1 -> CharbonnierLoss
    cleaning_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),  # per the paper, l1 -> CharbonnierLoss
    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={
            '2': 0.1,
            '7': 0.1,
            '16': 1.0,
            '25': 1.0,
            '34': 1.0,
        },
        vgg_type='vgg19',
        perceptual_weight=1.0,
        style_weight=0,
        norm_img=False),
    gan_loss=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=5e-2,
        real_label_val=1.0,
        fake_label_val=0),
    is_use_sharpened_gt_in_pixel=True,
    is_use_sharpened_gt_in_percep=True,
    is_use_sharpened_gt_in_gan=False,
    is_use_ema=True,
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

train_dataloader = dict(
    num_workers=10,
    batch_size=load_batch_size,  # must be even or 1
)

# optimizer
optim_wrapper = dict(
    _delete_=True,
    constructor='MultiOptimWrapperConstructor',
    generator=dict(
        type='OptimWrapper',
        accumulative_counts=5,  # gradient accumulation
        optimizer=dict(type='AdamW', lr=5e-5, betas=(0.9, 0.99), weight_decay=1e-6, eps=1e-08, amsgrad=True),
        clip_grad=dict(max_norm=1, norm_type=2),),
    discriminator=dict(
        type='OptimWrapper',
        accumulative_counts=5,  # gradient accumulation
        optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-6, eps=1e-08, amsgrad=True),
        clip_grad=dict(max_norm=1, norm_type=2),)
)

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=150_000, val_interval=2000)
