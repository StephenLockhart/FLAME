# Copyright (c) OpenMMLab. All rights reserved.
from .aimvrt import AimVRT  # Base AIM-VRT hybrid model
from .aimvrt_dynamic import AimVRTDynamic  # Dynamic Hilbert pipeline, supports arbitrary patch size
from .dmvrt_net_fixed import DMVRTNetFixed  # Fixed version: pa=2 single-interval optical flow + mask down-weighting
from .dmvrt_net_fixed_128 import DMVRTNetFixed128  # Variant optimized for 128x128 patches
from .dmvrt_net_fixed_160 import DMVRTNetFixed160  # Variant optimized for 160x160 patches

__all__ = [
    'AimVRT', 'AimVRTDynamic', 'DMVRTNetFixed', 'DMVRTNetFixed128',
    'DMVRTNetFixed160'
]
