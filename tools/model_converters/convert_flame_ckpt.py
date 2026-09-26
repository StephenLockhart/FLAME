# Copyright (c) OpenMMLab. All rights reserved.
"""Convert a raw FLAME training checkpoint to a clean inference checkpoint.

Training checkpoints produced by MMEngine may contain:
  - DDP-wrapped generator:    generator.module.xxx
  - DDP-wrapped EMA weights:  generator_ema.module.xxx
  - plain EMA weights:        generator_ema.xxx
  - optimizer/scheduler states and message history (large, unneeded)

This script keeps only the generator weights (EMA preferred when present).
Only the extra DDP ``module.`` layer introduced by DistributedDataParallel is
stripped; every output key keeps the ``generator.`` prefix so that the saved
``state_dict`` matches the ``AimVRT``/``AimVRTDynamic`` wrapper built by
``tools/test.py`` (a ``BaseEditModel`` whose network lives at
``model.generator``, i.e. keys ``generator.fre2.para1`` etc.). A compact
checkpoint is written that ``tools/test.py`` can consume directly.

Usage:
    python tools/model_converters/convert_flame_ckpt.py \
        work_dirs/xxx/best_PSNR_iter_160000.pth \
        checkpoints/flame_rainsynall100haze_psnr.pth
"""

import argparse
import os

import torch


def convert(in_path: str, out_path: str, prefer_ema: bool = True) -> None:
    ckpt = torch.load(in_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)

    def collect(prefix):
        # Some checkpoints wrap the EMA model in DDP as well, giving keys
        # such as ``generator_ema.module.xxx``; strip the extra ``module.``
        # layer but always keep the wrapper-level ``generator.`` prefix so
        # the state_dict matches AimVRT/AimVRTDynamic (model.generator).
        out = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                name = k[len(prefix):]
                if name.startswith('module.'):
                    name = name[len('module.'):]
                out[f'generator.{name}'] = v
        return out

    ema = collect('generator_ema.')
    ddp = collect('generator.module.')
    plain = collect('generator.')

    if prefer_ema and ema:
        source, gen = 'generator_ema', ema
    elif ddp:
        source, gen = 'generator.module (DDP)', ddp
    elif plain:
        source, gen = 'generator', plain
    else:
        raise RuntimeError(
            'No generator weights found. Keys seen: '
            f'{list(state_dict.keys())[:10]} ...')

    # Sanity check: output keys must be directly loadable by the wrapper.
    bad = [k for k in gen if not k.startswith('generator.')
           or k.startswith('generator.module.') or '.module.' in k]
    if bad:
        raise RuntimeError(
            f'Unexpected keys after conversion, e.g. {bad[:5]}')

    out = {'state_dict': gen}
    if 'meta' in ckpt:
        out['meta'] = ckpt['meta']

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(out, out_path)

    in_mb = os.path.getsize(in_path) / 1024**2
    out_mb = os.path.getsize(out_path) / 1024**2
    print(f'Source weights : {source} ({len(gen)} tensors)')
    print(f'Input          : {in_path} ({in_mb:.1f} MB)')
    print(f'Output         : {out_path} ({out_mb:.1f} MB)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('in_path', help='raw training checkpoint (.pth)')
    parser.add_argument('out_path', help='output clean checkpoint (.pth)')
    parser.add_argument(
        '--no-ema',
        action='store_true',
        help='use the raw generator weights instead of the EMA weights')
    args = parser.parse_args()
    convert(args.in_path, args.out_path, prefer_ema=not args.no_ema)
