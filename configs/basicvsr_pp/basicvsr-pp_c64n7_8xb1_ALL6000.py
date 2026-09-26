_base_ = '../basicvsr/basicvsr_2xb4_ALL6000.py'

load_batch_size = 8
load_input_frames = 10
name_input_frames = 2 * load_input_frames  # doubled because flip augmentation doubles the data
experiment_name = f'basicvsr-pp_c64n7_1x{name_input_frames}xb{load_batch_size}-300k_ALL6000'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs'

scale = 4
# load the checkpoint for faster adaptation

# model settings
model = dict(
    type='BasicVSR',
    generator=dict(
        type='BasicVSRPlusPlusNet',
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=True,
        spynet_pretrained='ckpt/spynet_20210409-c6c1bd09.pth',),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    train_cfg=dict(fix_iter=5000),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

train_dataloader = dict(
    num_workers=10, batch_size=load_batch_size, dataset=dict(num_input_frames=load_input_frames))

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=300_000, val_interval=2000)

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    accumulative_counts=5,  # gradient accumulation
    optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99)),
    paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.25)}),
    clip_grad=dict(max_norm=1, norm_type=2),  # norm-based gradient clipping
)

default_hooks = dict(checkpoint=dict(out_dir=save_dir))

# learning policy
param_scheduler = dict(
    type='CosineRestartLR',
    by_epoch=False,
    periods=[300000],
    restart_weights=[1],
    eta_min=1e-7)
