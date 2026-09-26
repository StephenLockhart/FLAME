# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from typing import Dict, List, Optional

from mmagic.registry import MODELS
from mmagic.structures import DataSample
from .aimvrt import AimVRT


@MODELS.register_module()
class AimVRTDynamic(AimVRT):
    """AimVRT pipeline supporting dynamic patch size

    Resolves the hard-coded Hilbert curve issue in AimVRT, supporting arbitrary patch sizes (64, 128, 256, 512...)

    Args:
        generator (dict): Config for the generator.
        pixel_loss (dict): Config for pixel-wise loss.
        perceptual_loss (dict, optional): Config for perceptual loss.
        contrast_loss (dict, optional): Config for contrast loss.
        train_cfg (dict): Config for training.
        test_cfg (dict): Config for testing.
        init_cfg (dict, optional): Config for initialization.
        data_preprocessor (dict, optional): Config for data preprocessor.
        scale_factor (int): Scale factor for video restoration, default 1.
        num_input_frames (int): Number of input frames, default 6.
    """

    def _init_hilbert_curves(self):
        """Dynamically generate Hilbert curves based on actual patch size

        Key fix:
        - Original AimVRT hard-codes large=64x64, small=32x32
        - 128x128 patch -> feature map 32x32 -> needs large=32x32, small=16x16
        - 256x256 patch -> feature map 64x64 -> needs large=64x64, small=32x32
        - Dynamic computation avoids tensor dimension mismatch errors
        """
        nf = self.num_input_frames

        # Dynamically generate Hilbert curves based on generator's img_size
        if hasattr(self.generator, 'img_size') and len(self.generator.img_size) >= 3:
            # img_size format: [T, H, W], e.g. [6, 64, 64] or [6, 32, 32]
            _, H, W = self.generator.img_size[:3]

            # Dynamically compute Hilbert curve sizes
            large_H, large_W = H, W
            small_H, small_W = max(H//2, 1), max(W//2, 1)

            print(f"AimVRTDynamic: Detected img_size={self.generator.img_size}")
            print(f"Dynamically generating Hilbert curves: large_scale={large_H}x{large_W}, small_scale={small_H}x{small_W}")

            # Generate Hilbert curves
            large_indices = self._create_hilbert_curve(H=large_H, W=large_W, nf=nf)
            small_indices = self._create_hilbert_curve(H=small_H, W=small_W, nf=nf)

            print(f"Successfully generated dynamic Hilbert curves: large={large_indices.shape}, small={small_indices.shape}")

            # Expected tensor length validation
            expected_large_len = nf * large_H * large_W
            expected_small_len = nf * small_H * small_W
            print(f"Expected tensor lengths: large={expected_large_len}, small={expected_small_len}")

        else:
            # Fallback: if img_size cannot be obtained, use original fixed values
            print("AimVRTDynamic: Cannot obtain img_size, falling back to fixed sizes 64x64, 32x32")
            large_indices = self._create_hilbert_curve(H=64, W=64, nf=nf)
            small_indices = self._create_hilbert_curve(H=32, W=32, nf=nf)

        # Use register_buffer to correctly register torch tensors
        self.register_buffer(
            "hilbert_large",
            torch.from_numpy(large_indices).long(),
            persistent=True
        )
        self.register_buffer(
            "hilbert_small",
            torch.from_numpy(small_indices).long(),
            persistent=True
        )

        print(f"Hilbert curves registered as module buffers, device will auto-sync")

    def forward_train(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[List[DataSample]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass, Hilbert curve device auto-syncs"""

        # Call parent's training method (device auto-sync)
        return super().forward_train(inputs, data_samples, **kwargs)

    def forward_inference(self, inputs, data_samples=None, **kwargs):
        """Inference forward pass, Hilbert curve device auto-syncs"""

        # Call parent's inference method (device auto-sync)
        return super().forward_inference(inputs, data_samples, **kwargs)

    def _test_video(self, lq, window_size, tile, tile_overlap, hilbert_large, hilbert_small):
        """Test video sequence using dynamic Hilbert curves"""

        # Ensure using module-registered Hilbert curves (device auto-sync)
        hilbert_large = self.hilbert_large
        hilbert_small = self.hilbert_small

        # Call parent method
        return super()._test_video(lq, window_size, tile, tile_overlap, hilbert_large, hilbert_small)

    def _test_clip(self, lq, window_size, tile, tile_overlap, hilbert_large, hilbert_small):
        """Test video clip using dynamic Hilbert curves"""

        # Ensure using module-registered Hilbert curves (device auto-sync)
        hilbert_large = self.hilbert_large
        hilbert_small = self.hilbert_small

        # Call parent method
        return super()._test_clip(lq, window_size, tile, tile_overlap, hilbert_large, hilbert_small)

    def check_if_mirror_extended(self, lrs):
        """Check mirror extension, device auto-syncs"""
        return super().check_if_mirror_extended(lrs)

    def sample_patches(
        self,
        output,
        gt,
        lq,
        scale=1,
        positive_range_initial=0,
        min_negative_distance_initial=96,
        patch_size=16,
        num_samples=10,
        overlap=False,
        step_counter_cl=0,
    ):
        """Sample patches, device auto-syncs"""

        return super().sample_patches(
            output, gt, lq, scale, positive_range_initial,
            min_negative_distance_initial, patch_size, num_samples,
            overlap, step_counter_cl
        )

    def forward_tensor(self, inputs, data_samples=None, **kwargs):
        """Tensor forward pass, device auto-syncs"""
        return super().forward_tensor(inputs, data_samples, **kwargs)

    def __repr__(self):
        """Return string representation of the class"""
        base_repr = super().__repr__()

        # Add dynamic Hilbert info
        if hasattr(self, 'hilbert_large') and hasattr(self, 'hilbert_small'):
            hilbert_info = f"\n  Dynamic Hilbert Curves: large={self.hilbert_large.shape}, small={self.hilbert_small.shape}"
            # Insert info before the last parenthesis
            base_repr = base_repr[:-1] + hilbert_info + base_repr[-1]

        return base_repr


# Export to main module for easier import
__all__ = ['AimVRTDynamic']
