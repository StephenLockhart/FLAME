# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
import numpy as np
from mmagic.structures import DataSample
from mmagic.models.base_models import BaseEditModel
from mmagic.models.editors.aimvsr.modules.Hilbert3d import Hilbert3d
from mmagic.registry import MODELS
from einops import rearrange


@MODELS.register_module()
class AimVSR(BaseEditModel):
    """Mamba-based video super-resolution model.

    Args:
        generator (dict): Config for the generator.
        pixel_loss (dict): Config for the pixel-wise loss.
        perceptual_loss (dict, optional): Config for the perceptual loss.
        contrast_loss (dict, optional): Config for the contrastive loss.
        train_cfg (dict): Config for training.
        test_cfg (dict): Config for testing.
        init_cfg (dict, optional): Config for initialization.
        data_preprocessor (dict, optional): Config for the data preprocessor.
    """

    def __init__(
        self,
        generator,
        pixel_loss,
        scale_factor=4,
        num_input_frames=5,
        perceptual_loss=None,
        contrast_loss=None,
        ensemble=None,
        train_cfg=None,
        test_cfg=None,
        init_cfg=None,
        data_preprocessor=None,
    ):
        super().__init__(
            generator=generator,
            pixel_loss=pixel_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor,
        )

        self.scale_factor = scale_factor
        self.num_input_frames = num_input_frames
        self.visualization_figure = None
        self.patches_count = 0

        # Initialize additional loss functions
        self.perceptual_loss = (
            MODELS.build(perceptual_loss) if perceptual_loss else None
        )
        self.contrast_loss = MODELS.build(contrast_loss) if contrast_loss else None

        # Initialize Hilbert curves
        self._init_hilbert_curves()

        # Initialize the training counter
        self.register_buffer("step_counter", torch.zeros(1))

        # Add contrastive learning related parameters following Derainer
        self.patch_size = 16  # Default patch size
        self.positive_range_initial = 0  # Initial positive sample range
        self.min_negative_distance_initial = 96  # Initial minimum negative distance
        self.num_samples = 10  # Number of samples per frame

        # Initialize ensemble
        self.forward_ensemble = self._init_ensemble(ensemble)

    def _init_ensemble(self, ensemble):
        """Initialize ensemble method if specified."""
        if ensemble is not None:
            if ensemble["type"] == "SpatialTemporalEnsemble":
                from mmagic.models.archs import SpatialTemporalEnsemble
                return SpatialTemporalEnsemble(
                    ensemble.get("is_temporal_ensemble", False)
                )
            raise NotImplementedError(f'Unsupported ensemble type: {ensemble["type"]}')
        return None

    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                is_mirror_extended = True

        return is_mirror_extended

    def _init_hilbert_curves(self):
        """Initialize Hilbert curves."""

        def _create_hilbert_curve(H, W, nf):
            hilbert_curve = list(Hilbert3d(width=H, height=W, depth=nf))
            hilbert_curve = torch.tensor(hilbert_curve).long()
            return (
                hilbert_curve[:, 0] * W * nf
                + hilbert_curve[:, 1] * nf
                + hilbert_curve[:, 2]
            )

        # Use the number of frames passed in during initialization
        nf = self.num_input_frames

        # Large-scale Hilbert curve (64x64)
        hilbert_large = _create_hilbert_curve(H=64, W=64, nf=nf)
        self.register_buffer("hilbert_large", hilbert_large, persistent=True)

        # Small-scale Hilbert curve (32x32)
        hilbert_small = _create_hilbert_curve(H=32, W=32, nf=nf)
        self.register_buffer("hilbert_small", hilbert_small, persistent=True)

    def sample_patches(
        self,
        output,
        gt,
        lq,
        scale=4,
        positive_range_initial=0,
        min_negative_distance_initial=96,
        patch_size=16,
        num_samples=10,
        overlap=False,
        step_counter_cl=0,
    ):
        """Sample patches for contrastive learning, handling different resolutions.

        Args:
            output (Tensor): Output tensor (B,T,C,H,W)
            gt (Tensor): GT tensor (B,T,C,H,W)
            lq (Tensor): LQ tensor (B,T,C,H//scale,W//scale)
        """
        B, T, C, H, W = output.shape

        # Upsample in two steps to avoid dimension issues
        # 1. Reshape into a batch of 2D images
        lq_reshaped = lq.view(B * T, C, H // scale, W // scale)

        # 2. Upsample with bicubic interpolation
        lq = F.interpolate(
            lq_reshaped,
            size=(H, W),
            mode="bicubic",
            align_corners=False
        ).view(B, T, C, H, W)

        # Get the training progress
        total_train = self.train_cfg.get("max_iters", 100000)
        progress_ratio = step_counter_cl / total_train
        decay_rate = 0.5

        # Use the same distance parameters as Derainer
        min_negative_distance = int(max(
            min_negative_distance_initial * (decay_rate ** progress_ratio),
            60
        ))
        positive_range = int(min(
            positive_range_initial + progress_ratio * (12 - positive_range_initial),
            12
        ))

        anchors, positives, negatives = [], [], []
        
        for b in range(B):
            for t in range(T):
                # Compute the difference map
                difference = torch.abs(gt[b, t] - lq[b, t])
                gray = difference.mean(dim=0) if C == 3 else difference
                threshold = gray.mean()
                binary = (gray > threshold).float()
                points = torch.nonzero(binary, as_tuple=False)

                for _ in range(num_samples):
                    # Sample the anchor
                    if len(points) == 0:
                        anchor_x = random.randint(0, W - patch_size)
                        anchor_y = random.randint(0, H - patch_size)
                    else:
                        point = points[random.randint(0, len(points) - 1)]
                        anchor_y = min(point[0].item(), H - patch_size)
                        anchor_x = min(point[1].item(), W - patch_size)
                    
                    anchor = output[b, t, :, 
                                  anchor_y:anchor_y+patch_size,
                                  anchor_x:anchor_x+patch_size]
                    anchors.append(anchor)

                    # Sample the positive
                    t_index = random.choice([t, max(0, t-1), min(T-1, t+1)])
                    if overlap:
                        positive_x = random.randint(
                            max(0, anchor_x - positive_range),
                            min(W - patch_size, anchor_x + positive_range)
                        )
                        positive_y = random.randint(
                            max(0, anchor_y - positive_range),
                            min(H - patch_size, anchor_y + positive_range)
                        )
                    else:
                        xdirection = random.choice(["left", "right"])
                        ydirection = random.choice(["up", "down"])
                        distance = random.randint(1, positive_range)
                        
                        if xdirection == "left":
                            positive_x = max(0, anchor_x - patch_size - distance)
                        else:
                            positive_x = min(W - patch_size, anchor_x + patch_size + distance)
                            
                        if ydirection == "up":
                            positive_y = max(0, anchor_y - patch_size - distance)
                        else:
                            positive_y = min(H - patch_size, anchor_y + patch_size + distance)

                    positive = gt[b, t_index, :,
                                positive_y:positive_y+patch_size,
                                positive_x:positive_x+patch_size]
                    positives.append(positive)

                    # Sample the negative
                    t_index_N = random.randint(0, T-1)
                    while True:
                        negative_x = random.randint(0, W - patch_size)
                        negative_y = random.randint(0, H - patch_size)
                        
                        if (abs(negative_x - anchor_x) > min_negative_distance or
                            abs(negative_y - anchor_y) > min_negative_distance):
                            
                            negative = lq[b, t_index_N, :,
                                        negative_y:negative_y+patch_size,
                                        negative_x:negative_x+patch_size]
                            
                            # Data augmentation
                            if random.random() < 0.5:
                                negative = transforms.ColorJitter(
                                    brightness=0.5,
                                    contrast=0.5,
                                    saturation=0.5,
                                    hue=0.25
                                )(negative)
                                
                            if random.random() < 0.5:
                                kernel_size = random.randint(1, 5)
                                if kernel_size % 2 == 0:
                                    kernel_size += 1
                                negative = transforms.GaussianBlur(
                                    kernel_size,
                                    sigma=(0.1, 2.0)
                                )(negative)
                                
                            negatives.append(negative)
                            break

        return (torch.stack(anchors),
                torch.stack(positives),
                torch.stack(negatives))

    def forward_train(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[List[DataSample]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward training with multiple losses.

        Args:
            inputs (torch.Tensor): Input tensor with shape (B, T, C, H, W)
            data_samples (List[DataSample], optional): Data samples

        Returns:
            Dict[str, torch.Tensor]: Dict of losses
        """
        # Forward through network
        output = self.forward_tensor(inputs, data_samples, **kwargs)
        batch_gt_data = data_samples.gt_img

        # b, t, c, h, w = output.size()

        # Calculate losses
        losses = dict()

        # 1. Pixel loss - keep 5D
        losses["loss_pix"] = self.pixel_loss(output, batch_gt_data)

        # 2. Perceptual loss - rearrange dimensions with einops for clarity
        if self.perceptual_loss:
            # [B,T,C,H,W] -> [B*T,C,H,W]
            outputs_percep = rearrange(output, "b t c h w -> (b t) c h w")
            gt_percep = rearrange(batch_gt_data, "b t c h w -> (b t) c h w")
            loss_percep, loss_style = self.perceptual_loss(outputs_percep, gt_percep)
            if loss_percep is not None:
                losses["loss_perceptual"] = loss_percep
            if loss_style is not None:
                losses["loss_style"] = loss_style

        # 3. Contrastive loss - keep 5D
        if self.contrast_loss:
            anchors, positives, negatives = self.sample_patches(
                output,
                batch_gt_data,
                inputs,
                scale=self.scale_factor,
                patch_size=self.patch_size,
                positive_range_initial=self.positive_range_initial,
                min_negative_distance_initial=self.min_negative_distance_initial,
                num_samples=self.num_samples,
                overlap=True,
                step_counter_cl=self.step_counter,
            )

            losses["loss_contrast"] = self.contrast_loss(anchors, positives, negatives)

        self.step_counter += 1
        return losses

    def forward_inference(self, inputs, data_samples=None, **kwargs):
        """Forward inference for validation/testing.
        Args:
            inputs (Tensor): Input tensor with shape [B, T, C, H, W]
            data_samples (List[DataSample]): List of DataSample
        Returns:
            DataSample: Output predictions
        """
        device = inputs.device
        hilbert_large = self.hilbert_large.to(device)
        hilbert_small = self.hilbert_small.to(device)

        # Use the config in test_cfg for tiled processing
        if self.test_cfg is not None:
            window_size = self.test_cfg.get('window_size', [2, 8, 8])
            tile = self.test_cfg.get('tile', [5, 256, 256])
            tile_overlap = self.test_cfg.get('tile_overlap', [4, 32, 32])
            
            output = self._test_video(
                inputs, 
                window_size=window_size,
                tile=tile,
                tile_overlap=tile_overlap,
                hilbert_large=hilbert_large,
                hilbert_small=hilbert_small
            )
        else:
            # Process the full input directly
            output = self.generator(inputs, hilbert_large, hilbert_small)

        # Process the output
        output = self.data_preprocessor.destruct(output, data_samples)

        # Create the prediction result
        predictions = DataSample(
            pred_img=output.cpu(),
            metainfo=data_samples.metainfo if data_samples is not None else None
        )

        return predictions

    def _test_video(self, lq, window_size, tile, tile_overlap, hilbert_large, hilbert_small):
        """Test video as a whole or as clips.
        Args:
            lq (Tensor): Input LR sequence with shape (n, t, c, h, w)
            window_size (list): Window size for processing
            tile (list): Tile size for splitting
            tile_overlap (list): Overlap size between tiles
            hilbert_large (Tensor): Large scale Hilbert curve
            hilbert_small (Tensor): Small scale Hilbert curve
        Returns:
            Tensor: Output HR sequence
        """
        num_frame_testing = tile[0]
        if num_frame_testing:
            # Get device information
            device = lq.device
            num_frame_overlap = tile_overlap[0]
            b, d, c, h, w = lq.size()
            stride = num_frame_testing - num_frame_overlap
            d_idx_list = list(range(0, d-num_frame_testing, stride)) + [max(0, d-num_frame_testing)]
            # Ensure E and W are on the correct device
            E = torch.zeros(b, d, c, h*self.scale_factor, w*self.scale_factor).to(device)
            W = torch.zeros(b, d, 1, 1, 1).to(device)

            for d_idx in d_idx_list:
                lq_clip = lq[:, d_idx:d_idx+num_frame_testing, ...]
                out_clip = self._test_clip(
                    lq_clip,
                    window_size=window_size,
                    tile=tile,
                    tile_overlap=tile_overlap,
                    hilbert_large=hilbert_large,
                    hilbert_small=hilbert_small
                )
                
                # # Show patches
                # plt.figure(figsize=(10, 5))
                # plt.subplot(121)
                # in_img = lq_clip[0,0].cpu().numpy()  # Use lq_clip as input
                # plt.imshow(in_img.transpose(1,2,0))  # Adjust dimension order
                # plt.title(f'Input clip frame 0')
                # plt.axis('off')
                
                # plt.subplot(122)
                # out_img = out_clip[0,0].cpu().numpy()  # Use out_clip as output
                # plt.imshow(out_img.transpose(1,2,0))  # Adjust dimension order
                # plt.title(f'Output clip frame 0 [{out_img.min():.2f}, {out_img.max():.2f}]')
                # plt.axis('off')
                # plt.show()
                
                # Ensure out_clip_mask is on the correct device
                out_clip_mask = torch.ones((b, min(num_frame_testing, d), 1, 1, 1)).to(device)

                # Handle overlapping regions
                if d_idx < d_idx_list[-1]:
                    out_clip[:, -num_frame_overlap//2:, ...] *= 0
                    out_clip_mask[:, -num_frame_overlap//2:, ...] *= 0
                if d_idx > d_idx_list[0]:
                    out_clip[:, :num_frame_overlap//2, ...] *= 0
                    out_clip_mask[:, :num_frame_overlap//2, ...] *= 0
                E[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip)
                W[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip_mask)
            output = E.div_(W)
            
        else:
            # Process the full video
            d_old = lq.size(1)
            d_pad = (window_size[0] - d_old % window_size[0]) % window_size[0]
            if d_pad:
                lq = torch.cat([lq, torch.flip(lq[:, -d_pad:, ...], [1])], 1)
            output = self._test_clip(
                lq,
                window_size=window_size,
                tile=tile,
                tile_overlap=tile_overlap,
                hilbert_large=hilbert_large,
                hilbert_small=hilbert_small
            )
            output = output[:, :d_old, :, :, :]

        return output

    def _test_clip(self, lq, window_size, tile, tile_overlap, hilbert_large, hilbert_small):
        """Test clip (a batch of images) using window splitting."""
        
        sf = self.scale_factor
        size_patch = tile[1]  # Use the patch size from the config
        overlap_size = tile_overlap[1]  # Use the overlap size from the config

        if size_patch:  # If tiled processing is used
            # Get input dimensions
            b, d, c, h, w = lq.size()
            device = lq.device  # Get the input device

            # Compute stride and index lists
            stride = size_patch - overlap_size
            h_idx_list = list(range(0, h-size_patch, stride)) + [max(0, h-size_patch)]
            w_idx_list = list(range(0, w-size_patch, stride)) + [max(0, w-size_patch)]
            
            # Create output tensor and weight tensor (ensure they are on the correct device)
            E = torch.zeros(b, d, c, h*sf, w*sf, device=device)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    # Extract the patch
                    in_patch = lq[..., h_idx:h_idx+size_patch, 
                                w_idx:w_idx+size_patch]
                    
                    # Process the patch
                    out_patch = self.generator(in_patch, hilbert_large, hilbert_small)
                    out_patch_mask = torch.ones_like(out_patch)

                    # Handle overlapping regions
                    if h_idx < h_idx_list[-1]:
                        out_patch[..., -overlap_size//2:, :] *= 0
                        out_patch_mask[..., -overlap_size//2:, :] *= 0
                    if w_idx < w_idx_list[-1]:
                        out_patch[..., :, -overlap_size//2:] *= 0
                        out_patch_mask[..., :, -overlap_size//2:] *= 0
                    if h_idx > h_idx_list[0]:
                        out_patch[..., :overlap_size//2, :] *= 0
                        out_patch_mask[..., :overlap_size//2, :] *= 0
                    if w_idx > w_idx_list[0]:
                        out_patch[..., :, :overlap_size//2] *= 0
                        out_patch_mask[..., :, :overlap_size//2] *= 0

                    # Ensure the patch is on the correct device
                    out_patch = out_patch.to(device)
                    out_patch_mask = out_patch_mask.to(device)

                    # Update the output
                    E[..., h_idx*sf:(h_idx+size_patch)*sf,
                      w_idx*sf:(w_idx+size_patch)*sf].add_(out_patch)
                    W[..., h_idx*sf:(h_idx+size_patch)*sf,
                      w_idx*sf:(w_idx+size_patch)*sf].add_(out_patch_mask)

            output = E.div_(W)

        else:  # If tiled processing is not used
            # Handle padding
            _, _, _, h_old, w_old = lq.size()
            h_pad = (window_size[1] - h_old % window_size[1]) % window_size[1]
            w_pad = (window_size[2] - w_old % window_size[2]) % window_size[2]

            if h_pad:
                lq = torch.cat([lq, torch.flip(lq[:, :, :, -h_pad:, :], [3])], 3)
            if w_pad:
                lq = torch.cat([lq, torch.flip(lq[:, :, :, :, -w_pad:], [4])], 4)

            # Process the full input directly
            output = self.generator(lq, hilbert_large, hilbert_small)
            output = output[:, :, :, :h_old*sf, :w_old*sf]

        return output

    def forward_tensor(self, inputs, data_samples=None, **kwargs):
        """Forward tensor.

        Args:
            inputs (torch.Tensor): Input tensor
            data_samples (List[DataSample], optional): Data samples

        Returns:
            torch.Tensor: Output tensor
        """
        # Forward through network
        return self.generator(inputs, self.hilbert_large, self.hilbert_small)
