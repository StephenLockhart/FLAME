# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
from mmengine import Config
from mmengine.registry import init_default_scope

from mmagic.registry import MODELS

try:
    from mmengine.analysis import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get a editor complexity',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[3, 250, 250],
        help='Input shape. Supported tasks:\n'
        'Image Super-Resolution: --shape 3 h w\n'
        'Video Super-Resolution: --shape t 3 h w\n'
        'Video Interpolation: --shape t 3 h w\n'
        'Image Restoration: --shape 3 h w\n'
        'Inpainting: --shape 4 h w\n'
        'Matting: --shape 4 h w\n'
        'Unconditional GANs: --shape noisy_size\n'
        'Image Translation: --shape 3 h w')
    parser.add_argument(
        '--activations',
        action='store_true',
        help='Whether to show the Activations')
    parser.add_argument(
        '--out-table',
        action='store_true',
        help='Whether to show the complexity table')
    parser.add_argument(
        '--out-arch',
        action='store_true',
        help='Whether to show the complexity arch')
    args = parser.parse_args()
    return args


def main():
    """
    Examples:

    Image Super-Resolution:
    `python tools/analysis_tools/get_flops.py configs/srcnn/srcnn_x4k915_1xb16-1000k_div2k.py --shape 3 250 250` # noqa

    Video Super-Resolution:
    `python tools/analysis_tools/get_flops.py configs/edvr/edvrm_8xb4-600k_reds.py --shape 5 3 256 256` # noqa

    Video Interpolation:
    `python tools/analysis_tools/get_flops.py configs/flavr/flavr_in4out1_8xb4_vimeo90k-septuplet.py --shape 4 3 64 64` # noqa

    Image Restoration:
    `python tools/analysis_tools/get_flops.py configs/nafnet/nafnet_c64eb11128mb1db1111_8xb8-lr1e-3-400k_gopro.py --shape 3 128 128` # noqa

    Inpainting:
    `python tools/analysis_tools/get_flops.py configs/aot_gan/aot-gan_smpgan_4xb4_places-512x512.py --shape 4 64 64` # noqa

    Matting:
    `python tools/analysis_tools/get_flops.py configs/dim/dim_stage1-v16_1xb1-1000k_comp1k.py --shape 4 256 256` # noqa

    Unconditional GANs:
    `python tools/analysis_tools/get_flops.py configs/wgan-gp/wgangp_GN_1xb64-160kiters_celeba-cropped-128x128.py --shape 128` # noqa

    Image Translation:
    `python tools/analysis_tools/get_flops.py configs/cyclegan/cyclegan_lsgan-id0-resnet-in_1xb1-250kiters_summer2winter.py --shape 3 250 250`
    """

    args = parse_args()

    input_shape = tuple(args.shape)

    cfg = Config.fromfile(args.config)

    init_default_scope(cfg.get('default_scope', 'mmagic'))

    model = MODELS.build(cfg.model)
    
    # get the number of frames from the input shape
    nf = input_shape[0]  # the first dimension is the frame count
    
    # create the Hilbert curve generator
    def _create_hilbert_curve(H, W, nf):
        from mmagic.models.editors.aimvsr.modules.Hilbert3d import Hilbert3d
        hilbert_curve = list(Hilbert3d(width=H, height=W, depth=nf))
        hilbert_curve = torch.tensor(hilbert_curve).long()
        return (
            hilbert_curve[:, 0] * W * nf
            + hilbert_curve[:, 1] * nf
            + hilbert_curve[:, 2]
        )
    
    # generate Hilbert curves at three different scales
    # hilbert_curve_super_large_scale = _create_hilbert_curve(H=256, W=256, nf=nf)
    # hilbert_curve_large_scale = _create_hilbert_curve(H=64, W=64, nf=nf)
    # hilbert_curve_small_scale = _create_hilbert_curve(H=32, W=32, nf=nf)
    # derive the feature-map resolution from the input size
    _, _, input_h, input_w = input_shape if len(input_shape) == 4 else (None, None, None, None)

    # defaults: consistent with training, corresponding to 256 -> 64 -> 32
    h_super = input_h if input_h is not None else 256
    w_super = input_w if input_w is not None else 256
    h_large = h_super // 4
    w_large = w_super // 4
    h_small = h_super // 8
    w_small = w_super // 8

    # LocalMambaBlock requires H=W (64/32 by default); otherwise restore the defaults to avoid reshape errors
    if h_large != w_large:
        h_large = w_large = 64
    if h_small != w_small:
        h_small = w_small = 32

    hilbert_curve_super_large_scale = _create_hilbert_curve(H=h_super, W=w_super, nf=nf)
    hilbert_curve_large_scale = _create_hilbert_curve(H=h_large, W=w_large, nf=nf)
    hilbert_curve_small_scale = _create_hilbert_curve(H=h_small, W=w_small, nf=nf)
    
    # combine all inputs into a tuple
    inputs = (torch.randn(1, *input_shape), 
            #  hilbert_curve_super_large_scale,   # used for the super-large scale
             hilbert_curve_large_scale, 
             hilbert_curve_small_scale)

    if torch.cuda.is_available():
        model.cuda()
        inputs = tuple(x.cuda() for x in inputs)
    model.eval()

    if hasattr(model, 'generator'):
        model = model.generator
    elif hasattr(model, 'backbone'):
        model = model.backbone
    if hasattr(model, 'translation'):
        model.forward = model.translation
    elif hasattr(model, 'infer'):
        model.forward = model.infer

    analysis_results = get_model_complexity_info(model, inputs=inputs)
    flops = analysis_results['flops_str']
    params = analysis_results['params_str']
    activations = analysis_results['activations_str']

    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}\n')
    if args.activations:
        print(f'Activations: {activations}\n{split_line}\n')
    if args.out_table:
        print(analysis_results['out_table'], '\n')
    if args.out_arch:
        print(analysis_results['out_arch'], '\n')

    if len(input_shape) == 4:
        print('!!!If your network computes N frames in one forward pass, you '
              'may want to divide the FLOPs by N to get the average FLOPs '
              'for each frame.')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
