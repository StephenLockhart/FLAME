# Third-Party Notices

FLAME is built on top of several excellent open-source projects. This file
records their provenance and licenses, which also apply to the corresponding
files in this repository.

## 1. MMagic framework (Apache License 2.0)

The overall framework (`mmagic/`, `tools/`, `configs/`, etc.) is derived from
[OpenMMLab MMagic](https://github.com/open-mmlab/mmagic), licensed under the
Apache License 2.0. The full license text is in [LICENSE](LICENSE).

> Copyright (c) OpenMMLab. All rights reserved.

## 2. VRT components (CC BY-NC 4.0)

`mmagic/models/editors/flame/VRT-main/` contains code adapted from
[VRT: Video Restoration Transformer](https://github.com/JingyunLiang/VRT)
(Jingyun Liang et al.), which is licensed under the
**Creative Commons Attribution-NonCommercial 4.0 International License
(CC BY-NC 4.0)**. The complete license text is kept in
`mmagic/models/editors/flame/VRT-main/LICENSE`.

Files in that directory (including `models/network_vrt.py`, which provides
`SpyNet`, `Stage`, `DCNv2PackFlowGuided`, and `Mlp_GEGLU`) additionally retain
upstream copyright notices (e.g., Facebook, Inc. / xformers, BSD-licensed;
BasicSR, Apache-2.0-licensed).

**These components are for non-commercial research use only.** They are not
covered by the Apache License 2.0 of the rest of this repository. Commercial
use requires permission from the original VRT authors. Modifications made for
FLAME: platform-safe SpyNet weight downloading (Windows compatibility).

The pretrained SpyNet weights (`spynet_sintel_final-3d2a1287.pth`) are
downloaded from the official VRT release page and are subject to the same
non-commercial research terms.

## 3. Gilbert space-filling curve (BSD 2-Clause)

`mmagic/models/editors/aimvsr/modules/Hilbert3d.py` is
Copyright (c) 2018 Jakub Červený, from
https://github.com/jakubcerveny/gilbert, licensed under the BSD 2-Clause
"Software" License (SPDX header retained in the file).

## 4. OpenMMLab components (Apache License 2.0)

- `mmagic/models/editors/aimvsr/modules/convnext.py` — adapted from
  [mmclassification ConvNeXt](https://github.com/open-mmlab/mmclassification)
  (Copyright (c) OpenMMLab).
- `mmagic/models/editors/aimvsr/modules/basicvr_pp_net.py`, and the utilities
  `PixelShufflePack`, `ResidualBlockNoBN`, `flow_warp`, `make_layer` —
  adapted from OpenMMLab MMagic/BasicVSR++ (Copyright (c) OpenMMLab).
- `mmagic/models/editors/aimvsr/modules/head.py` — FPN-style projection head
  following the OpenMMLab/mmdetection design pattern.

## 5. Mamba / timm building blocks (Apache License 2.0)

`mmagic/models/editors/aimvsr/modules/mambablock.py` and `Hymamba.py` contain
FLAME-specific 3D extensions built upon:

- [state-spaces/mamba](https://github.com/state-spaces/mamba) (Apache 2.0)
- [huggingface/pytorch-image-models (timm)](https://github.com/huggingface/pytorch-image-models) (Apache 2.0)
- [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models) utilities referenced inside VRT code

## 6. pytorch_diffusion (MIT)

`mmagic/models/editors/aimvsr/modules/ResNet.py` is derived from
[pytorch_diffusion](https://github.com/pesser/pytorch_diffusion) /
[CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
encoder-decoder code, as noted in the file header.

## 7. Pretrained weights

- **ConvNeXt-Tiny** encoder weights: downloaded from the official OpenMMLab
  model zoo (`download.openmmlab.com`), Apache-2.0-compatible model license.
- **VGG16** weights (training-time perceptual loss only): provided by
  torchvision, for research use.
- **FLAME checkpoints**: released for non-commercial academic research use.

---

If you are the author of any component listed above and believe the attribution
is inaccurate, please open an issue and we will correct it promptly.
