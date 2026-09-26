# Copyright (c) OpenMMLab. All rights reserved.
from .flame import Flame  # Base FLAME hybrid model
from .flame_dynamic import FlameDynamic  # Dynamic Hilbert pipeline, supports arbitrary patch size
from .flame_net import FlameNet  # Fixed version: pa=2 single-interval optical flow + mask down-weighting
from .flame_net_128 import FlameNet128  # Variant optimized for 128x128 patches
from .flame_net_160 import FlameNet160  # Variant optimized for 160x160 patches

# legacy aliases for checkpoint compatibility
AimVRT = Flame
AimVRTDynamic = FlameDynamic
DMVRTNetFixed = FlameNet
DMVRTNetFixed128 = FlameNet128
DMVRTNetFixed160 = FlameNet160

__all__ = [
    'Flame', 'FlameDynamic', 'FlameNet', 'FlameNet128', 'FlameNet160',
    # legacy aliases for checkpoint compatibility
    'AimVRT', 'AimVRTDynamic', 'DMVRTNetFixed', 'DMVRTNetFixed128',
    'DMVRTNetFixed160'
]
