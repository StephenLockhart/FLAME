_base_ = '../basicvsr/basicvsr_2xb4_ALL6000.py'

load_batch_size = 8
load_input_frames = 10
name_input_frames = 2 * load_input_frames
experiment_name = f'iconvsr_1x{name_input_frames}xb{load_batch_size}_ALL6000'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# model settings
model = dict(
    type='BasicVSR',
    generator=dict(
        type='IconVSRNet',
        mid_channels=64,
        num_blocks=30,
        keyframe_stride=5,
        padding=2,
        spynet_pretrained='ckpt/spynet_20210409-c6c1bd09.pth',
        edvr_pretrained='ckpt/edvrm_reds_20210413-3867262f.pth'),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    train_cfg=dict(fix_iter=5000),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

train_dataloader = dict(
    num_workers=10,
    batch_size=load_batch_size,  # must be even or 1
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(num_input_frames=load_input_frames,  # 15 change to load_input_frames
                 )
)
