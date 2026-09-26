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
import math  # added at the top of the file


@MODELS.register_module()
class AimVR(BaseEditModel):                                 # Video restoration model based on RainMamba, with fixed contrastive loss
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

    # Class-level CPU cache
    _CPU_HILBERT_CACHE = {}
    
    @staticmethod
    def _create_hilbert_curve(H: int, W: int, nf: int) -> np.ndarray:
        """Create a 3D Hilbert curve on CPU.

        Args:
            H (int): Height.
            W (int): Width.
            nf (int): Number of frames.

        Returns:
            np.ndarray: Hilbert curve indices.
        """
        cache_key = f"{H}_{W}_{nf}"
        
        # Check the CPU cache
        if cache_key not in AimVR._CPU_HILBERT_CACHE:
            # Create the curve with numpy
            hilbert_points = np.array(list(Hilbert3d(width=H, height=W, depth=nf)))
            # Pre-compute the indices
            indices = (hilbert_points[:, 0] * W * nf + 
                      hilbert_points[:, 1] * nf + 
                      hilbert_points[:, 2])
            # Store in the CPU cache
            AimVR._CPU_HILBERT_CACHE[cache_key] = indices
            
        return AimVR._CPU_HILBERT_CACHE[cache_key]

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
        self.min_negative_distance_initial = 96  # Initial minimum negative distance, reduced but fixed
        self.num_samples = 4  # Number of samples across the whole batch each time, consistent with Derainer

        # Initialize ensemble
        self.forward_ensemble = self._init_ensemble(ensemble)

        # Pre-define transforms and move them to GPU
        self.color_jitter = transforms.ColorJitter(
            brightness=0.5, contrast=0.5,
            saturation=0.5, hue=0.25
        )
        self.gaussian_blur = transforms.GaussianBlur(3, sigma=(0.1, 2.0))

        # # Statistics-related initialization
        # self.sampling_stats = {
        #     'anchor_from_strict': 0,    
        #     'anchor_from_mean': 0,      
        #     'anchor_random': 0,         
        #     'negative_from_batch': 0,   
        #     'negative_from_points': 0,  
        #     'negative_random': 0,       
        #     'total_samples': 0,
        #     'random_attempts': []
        # }

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
        """Initialize Hilbert curves and move them to GPU."""
        nf = self.num_input_frames
        
        # Get curves from the CPU cache
        large_indices = self._create_hilbert_curve(H=64, W=64, nf=nf)
        small_indices = self._create_hilbert_curve(H=32, W=32, nf=nf)
        
        # Create directly as CUDA tensors and register as buffers
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
        # print(f"\nInput shapes:")
        # print(f"output: {output.shape}")
        # print(f"gt: {gt.shape}")
        # print(f"lq: {lq.shape}")
        
        device = output.device
        # Upsample in two steps to avoid dimension issues
        # 1. Reshape into a batch of 2D images
        lq_reshaped = lq.view(B * T, C, H // scale, W // scale)
        # print(f"lq_reshaped: {lq_reshaped.shape}")

        if scale != 1:
            lq = F.interpolate(
                lq_reshaped,
                size=(H, W),
                mode="bicubic",
                align_corners=False
            ).view(B, T, C, H, W)
            # print(f"lq after interpolate: {lq.shape}")

        # Get the training progress
        total_train = self.train_cfg.get("max_iters", 100000)
        progress_ratio = step_counter_cl / total_train
        decay_rate = 0.5

        # Use the same distance parameters as Derainer
        min_negative_distance = int(max(
            min_negative_distance_initial * (decay_rate ** progress_ratio),
            64
        ))
        positive_range = int(min(
            positive_range_initial + progress_ratio * (8 - positive_range_initial),
            8
        ))
        num_samples = int(num_samples / B)  # Keep the total number of samples unchanged, adjusted by batch size
        device = output.device  # Get the current device

        # Pre-create data augmentation operations
        color_jitter = self.color_jitter.to(device)
        gaussian_blur = self.gaussian_blur.to(device)
        
        anchors, positives, negatives = [], [], []
        
        # # Reset statistics for this sampling round
        # batch_stats = {
        #     'anchor_from_strict': 0,
        #     'anchor_from_mean': 0,
        #     'anchor_random': 0,
        #     'negative_from_batch': 0,
        #     'negative_from_points': 0,
        #     'negative_random': 0,
        #     'total_samples': 0
        # }
        
        for b in range(B):
            t = torch.randint(0, T, (1,), device=device).item()
            # print(f"\nProcessing batch {b}, time {t}")
                
            # Compute the difference map (already on CUDA)
            difference = torch.abs(gt[b, t] - lq[b, t])
            gray = difference.mean(dim=0) if C == 3 else difference
            
            # Compute thresholds (already on CUDA)
            mean_threshold = gray.mean()
            strict_threshold = mean_threshold + gray.std()
            
            # Get candidate points (already on CUDA)
            points_anchor = torch.nonzero((gray > strict_threshold).float())
            points_pos = torch.nonzero((gray > mean_threshold).float())

            for _ in range(num_samples):
                # Comment out all statistics-related code
                # batch_stats['total_samples'] += 1
                # if len(points_anchor) > 0:
                #     batch_stats['anchor_from_strict'] += 1
                # elif len(points_pos) > 0:
                #     batch_stats['anchor_from_mean'] += 1
                # else:
                #     batch_stats['anchor_random'] += 1
                
                # Sample the anchor
                if len(points_anchor) > 0:
                    idx = torch.randint(0, len(points_anchor), (1,), device=device)
                    # Get the y and x coordinates
                    anchor_y = min(points_anchor[idx, 0].item(), H - patch_size)
                    anchor_x = min(points_anchor[idx, 1].item(), W - patch_size)
                    # if found_negative:
                    #     if use_other_batch:
                    #         batch_stats['negative_from_batch'] += 1
                    #     else:
                    #         batch_stats['negative_from_points'] += 1
                    # else:
                    #     batch_stats['negative_random'] += 1
                elif len(points_pos) > 0:
                    idx = torch.randint(0, len(points_pos), (1,), device=device)
                    anchor_y = min(points_pos[idx, 0].item(), H - patch_size)
                    anchor_x = min(points_pos[idx, 1].item(), W - patch_size)
                    # if found_negative:
                    #     if use_other_batch:
                    #         batch_stats['negative_from_batch'] += 1
                    #     else:
                    #         batch_stats['negative_from_points'] += 1
                    # else:
                    #     batch_stats['negative_random'] += 1
                else:
                    anchor_x = torch.randint(0, W - patch_size, (1,), device=device).item()
                    anchor_y = torch.randint(0, H - patch_size, (1,), device=device).item()
                    # if found_negative:
                    #     batch_stats['negative_random'] += 1
                
                # Get the anchor patch (already on CUDA)
                anchor = output[b, t, :, 
                                anchor_y:anchor_y+patch_size,
                                anchor_x:anchor_x+patch_size]
                # print(f"anchor patch: {anchor.shape}")
                
                # Sample the positive
                t_candidates = torch.tensor([t, max(0, t-1), min(T-1, t+1)], device=device)
                t_index = t_candidates[torch.randint(0, len(t_candidates), (1,), device=device)].item()
                
                if overlap:
                    positive_x = torch.randint(
                        max(0, anchor_x - positive_range),
                        min(W - patch_size, anchor_x + positive_range) + 1,
                        (1,), device=device
                    ).item()
                    positive_y = torch.randint(
                        max(0, anchor_y - positive_range),
                        min(H - patch_size, anchor_y + positive_range) + 1,
                        (1,), device=device
                    ).item()
                else:
                    # Use torch.randint instead of random.choice
                    xdirection = torch.randint(0, 2, (1,), device=device).item()  # 0:left, 1:right
                    ydirection = torch.randint(0, 2, (1,), device=device).item()  # 0:up, 1:down
                    distance = torch.randint(1, positive_range + 1, (1,), device=device).item()
                    
                    if xdirection == 0:  # left
                        positive_x = max(0, anchor_x - patch_size - distance)
                    else:  # right
                        positive_x = min(W - patch_size, anchor_x + patch_size + distance)
                        
                    if ydirection == 0:  # up
                        positive_y = max(0, anchor_y - patch_size - distance)
                    else:  # down
                        positive_y = min(H - patch_size, anchor_y + patch_size + distance)

                positive = gt[b, t_index, :,
                            positive_y:positive_y+patch_size,
                            positive_x:positive_x+patch_size]
                # print(f"positive patch: {positive.shape}")

                # Sample the negative
                found_negative = False
                use_other_batch = torch.rand(1, device=device).item() < 0.2
                
                if use_other_batch and B > 1:
                    # Randomly select from other batches
                    other_b_candidates = [i for i in range(B) if i != b]
                    other_b = other_b_candidates[torch.randint(0, len(other_b_candidates), (1,), device=device).item()]
                    
                    # Randomly select 5 points in the same frame t and take the one with the largest difference
                    max_diff = -float('inf')
                    best_x = None
                    best_y = None
                    
                    for _ in range(5):
                        neg_y = torch.randint(0, H - patch_size, (1,), device=device).item()
                        neg_x = torch.randint(0, W - patch_size, (1,), device=device).item()
                        
                        # Compute the difference value
                        curr_diff = torch.abs(gt[other_b, t] - lq[other_b, t]).mean(dim=0)[neg_y, neg_x]
                        
                        if curr_diff > max_diff:
                            max_diff = curr_diff
                            best_x = neg_x
                            best_y = neg_y
                    
                    # Set found_negative after finding the point with maximum difference
                    found_negative = True
                    negative = lq[other_b, t, :,
                                    best_y:best_y+patch_size,
                                    best_x:best_x+patch_size]
                    # if found_negative:
                    #     if use_other_batch:
                    #         batch_stats['negative_from_batch'] += 1
                    #     else:
                    #         batch_stats['negative_from_points'] += 1
                    # else:
                    #     batch_stats['negative_random'] += 1

                # # First try to find a suitable point from points_pos; almost 100% are difference points
                # if len(points_pos) > 0:
                #     # Randomly shuffle the candidate points
                #     candidates = points_pos[torch.randperm(len(points_pos))]
                    
                #     for point in candidates:
                #         neg_y = max(0, min(point[0].item(), H - patch_size))
                #         neg_x = max(0, min(point[1].item(), W - patch_size))
                        
                #         # Any point satisfying the distance requirement can serve as a negative sample
                #         if (abs(neg_x - anchor_x) > min_negative_distance or
                #             abs(neg_y - anchor_y) > min_negative_distance):
                #             negative = lq[b, t, :,
                #                             neg_y:neg_y+patch_size,
                #                             neg_x:neg_x+patch_size]
                #             found_negative = True
                #             batch_stats['negative_from_points'] += 1
                #             break
                
                # If not sampled from other batches, try sampling from points_pos; 30% of the time points_pos sampling is skipped directly
                if not found_negative:
                    use_points = torch.rand(1, device=device).item() < 0.7  # Add probability control
                    if use_points and len(points_pos) > 0:
                        candidates = points_pos[torch.randperm(len(points_pos))]
                        
                        for point in candidates:
                            neg_y = max(0, min(point[0].item(), H - patch_size))
                            neg_x = max(0, min(point[1].item(), W - patch_size))
                            
                            if (abs(neg_x - anchor_x) > min_negative_distance or
                                abs(neg_y - anchor_y) > min_negative_distance):
                                negative = lq[b, t, :,
                                                neg_y:neg_y+patch_size,
                                                neg_x:neg_x+patch_size]
                                found_negative = True
                                # if found_negative:
                                #     if use_other_batch:
                                #         batch_stats['negative_from_batch'] += 1
                                #     else:
                                #         batch_stats['negative_from_points'] += 1
                                # else:
                                #     batch_stats['negative_random'] += 1
                                break

                # If no suitable negative sample is found, try random sampling
                if not found_negative:
                    attempt_count = 0
                    max_attempts = 5  # Maximum number of attempts
                    
                    for attempt in range(max_attempts):
                        attempt_count += 1
                        negative_x = torch.randint(0, W - patch_size, (1,), device=device).item()
                        negative_y = torch.randint(0, H - patch_size, (1,), device=device).item()
                        
                        if (abs(negative_x - anchor_x) > min_negative_distance or
                            abs(negative_y - anchor_y) > min_negative_distance):
                            negative = lq[b, t, :,
                                        negative_y:negative_y+patch_size,
                                        negative_x:negative_x+patch_size]
                            found_negative = True
                            # if found_negative:
                            #     batch_stats['negative_random'] += 1
                            # else:
                            #     batch_stats['negative_random'] += 1
                            break
                    
                    # If no suitable point is found after 5 attempts, choose randomly
                    if not found_negative:
                        negative_x = torch.randint(0, W - patch_size, (1,), device=device).item()
                        negative_y = torch.randint(0, H - patch_size, (1,), device=device).item()
                        negative = lq[b, t, :,
                                    negative_y:negative_y+patch_size,
                                    negative_x:negative_x+patch_size]
                        # if found_negative:
                        #     batch_stats['negative_random'] += 1
                        # else:
                        #     batch_stats['negative_random'] += 1

                # Data augmentation
                if torch.rand(1, device=device).item() < 0.5:
                    # ColorJitter needs to run on CPU, but we can optimize the data transfer
                    # negative_cpu = negative.cpu()
                    # negative_cpu = self.color_jitter(negative_cpu)
                    # negative = negative_cpu.to(device)
                    negative = color_jitter(negative)
                    # print(f"negative after color jitter: {negative.shape}")
                
                if torch.rand(1, device=device).item() < 0.5:
                    # GaussianBlur also needs to run on CPU
                    # negative_cpu = negative.cpu()
                    # negative_cpu = self.gaussian_blur(negative_cpu)
                    # negative = negative_cpu.to(device)
                    negative = gaussian_blur(negative)
                    # print(f"negative after gaussian blur: {negative.shape}")
                
                # Geometric transforms can be performed directly on GPU
                if torch.rand(1, device=device).item() < 0.5:
                    # Use torch.randint instead of random.choice
                    transform_type = torch.randint(0, 5, (1,), device=device).item()
                    
                    if transform_type == 0:  # flip_h
                        negative = torch.flip(negative, [-1])
                    elif transform_type == 1:  # flip_v
                        negative = torch.flip(negative, [-2])
                    else:  # rot90, rot180, rot270
                        k = transform_type - 1  # 1->90°, 2->180°, 3->270°
                        negative = torch.rot90(negative, k=k, dims=[-2, -1])
                    # print(f"negative after geometric transform: {negative.shape}")
                # Append directly to the list; no extra dimension check needed
                anchors.append(anchor)
                # print(f"anchors_list length: {len(anchors)}")
                positives.append(positive)
                # print(f"positives_list length: {len(positives)}")
                negatives.append(negative)
                # print(f"negatives_list length: {len(negatives)}")

        # Stack operation
        anchors = torch.stack(anchors)
        positives = torch.stack(positives)
        negatives = torch.stack(negatives)
        # print(f"\nAfter stacking:")
        # print(f"anchors: {anchors.shape}")
        # print(f"positives: {positives.shape}")
        # print(f"negatives: {negatives.shape}")
        
        # Ensure the dimensions are correct [N*T, C, H, W]
        if anchors.dim() == 5:
            N, T, C, H, W = anchors.shape
            anchors = anchors.view(-1, C, H, W)
            positives = positives.view(-1, C, H, W)
            negatives = negatives.view(-1, C, H, W)
            # print(f"\nFinal shapes after reshaping:")
            # print(f"anchors: {anchors.shape}")
            # print(f"positives: {positives.shape}")
            # print(f"negatives: {negatives.shape}")
        
        # # Comment out statistics update
        # for key in batch_stats:
        #     if key != 'random_attempts':
        #         self.sampling_stats[key] += batch_stats[key]
        
        # # Comment out statistics printing
        # if self.sampling_stats['total_samples'] % 100 == 0:
        #     print("\n=== Sampling Statistics ===")
        #     print(f"Total samples: {self.sampling_stats['total_samples']}")
        #     print("\nAnchor sampling:")
        #     print(f"- From strict threshold: {self.sampling_stats['anchor_from_strict']} "
        #           f"({self.sampling_stats['anchor_from_strict']/self.sampling_stats['total_samples']*100:.1f}%)")
        #     print(f"- From mean threshold: {self.sampling_stats['anchor_from_mean']} "
        #           f"({self.sampling_stats['anchor_from_mean']/self.sampling_stats['total_samples']*100:.1f}%)")
        #     print(f"- Random sampling: {self.sampling_stats['anchor_random']} "
        #           f"({self.sampling_stats['anchor_random']/self.sampling_stats['total_samples']*100:.1f}%)")
        #     print("\nNegative sampling:")
        #     print(f"- From other batches: {self.sampling_stats['negative_from_batch']} "
        #           f"({self.sampling_stats['negative_from_batch']/self.sampling_stats['total_samples']*100:.1f}%)")
        #     print(f"- From difference points: {self.sampling_stats['negative_from_points']} "
        #           f"({self.sampling_stats['negative_from_points']/self.sampling_stats['total_samples']*100:.1f}%)")
        #     print(f"- Random sampling: {self.sampling_stats['negative_random']} "
        #           f"({self.sampling_stats['negative_random']/self.sampling_stats['total_samples']*100:.1f}%)")
            
        #     # Print only when random sampling data exists
        #     if len(self.sampling_stats['random_attempts']) > 0:
        #         attempts = self.sampling_stats['random_attempts']
        #         avg_attempts = sum(attempts) / len(attempts)
        #         print("\nRandom sampling efficiency:")
        #         print(f"- Average attempts needed: {avg_attempts:.2f}")
        #     print("========================\n")
        
        return (anchors, positives, negatives)

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
                num_samples=self.num_samples,  # Dynamically adjusted by batch size
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
        
        # Ensure Hilbert curves are on the correct device
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
        """Test video as a whole or as clips."""
        # Get transition settings from test_cfg; hard transition by default
        use_temporal_gradient = self.test_cfg.get('use_temporal_gradient', False)  # Gaussian weights
        use_temporal_average = self.test_cfg.get('use_temporal_average', False)   # Cumulative average
        
        if use_temporal_gradient:
            print("Using temporal smooth transition with gaussian weights")
        elif use_temporal_average:
            print("Using temporal smooth transition with cumulative average") 
        else:
            print("Using temporal hard transition")

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
                out_clip = self._test_clip(lq_clip, window_size, tile, tile_overlap, 
                                         hilbert_large, hilbert_small)
                out_clip_mask = torch.ones((b, min(num_frame_testing, d), 1, 1, 1)).to(device)

                # Handle overlapping regions according to the transition mode
                if use_temporal_gradient:  # Gaussian weights
                    if d_idx < d_idx_list[-1]:
                        weight = self._create_temporal_weight(num_frame_overlap//2).to(device)
                        out_clip[:, -num_frame_overlap//2:, ...] *= weight.view(-1, 1, 1, 1)
                        out_clip_mask[:, -num_frame_overlap//2:, ...] *= weight.view(-1, 1, 1, 1)
                    if d_idx > d_idx_list[0]:
                        weight = self._create_temporal_weight(num_frame_overlap//2).flip(0).to(device)
                        out_clip[:, :num_frame_overlap//2, ...] *= weight.view(-1, 1, 1, 1)
                        out_clip_mask[:, :num_frame_overlap//2, ...] *= weight.view(-1, 1, 1, 1)
                elif use_temporal_average:  # Cumulative average
                    # No processing needed, accumulate directly
                    pass
                else:  # Hard transition
                    if d_idx < d_idx_list[-1]:
                        out_clip[:, -num_frame_overlap//2:, ...] *= 0
                        out_clip_mask[:, -num_frame_overlap//2:, ...] *= 0
                    if d_idx > d_idx_list[0]:
                        out_clip[:, :num_frame_overlap//2, ...] *= 0
                        out_clip_mask[:, :num_frame_overlap//2, ...] *= 0

                # Accumulate into the output
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
        """Test clip using window splitting."""
        # Get spatial transition settings from test_cfg
        use_spatial_gradient = self.test_cfg.get('use_spatial_gradient', False)  # Gaussian weights
        use_spatial_average = self.test_cfg.get('use_spatial_average', False)   # Cumulative average
        
        if use_spatial_gradient:
            print("Using spatial smooth transition with gaussian weights")
        elif use_spatial_average:
            print("Using spatial smooth transition with cumulative average")
        else:
            print("Using spatial hard transition")

        sf = self.scale_factor
        size_patch = tile[1]
        overlap_size = tile_overlap[1]
        
        if size_patch:
            device = lq.device
            b, d, c, h, w = lq.size()
            stride = size_patch - overlap_size
            h_idx_list = list(range(0, h-size_patch, stride)) + [max(0, h-size_patch)]
            w_idx_list = list(range(0, w-size_patch, stride)) + [max(0, w-size_patch)]
            
            E = torch.zeros(b, d, c, h*sf, w*sf).to(device)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = lq[..., h_idx:h_idx+size_patch, w_idx:w_idx+size_patch]
                    out_patch = self.generator(in_patch, hilbert_large, hilbert_small)
                    out_patch_mask = torch.ones_like(out_patch)

                    # Handle overlapping regions according to the transition mode
                    if use_spatial_gradient:  # Gaussian weights
                        if h_idx < h_idx_list[-1]:
                            w = self._create_spatial_weight(overlap_size//2).to(device)
                            out_patch[..., -overlap_size//2:, :] *= w.view(-1, 1)
                            out_patch_mask[..., -overlap_size//2:, :] *= w.view(-1, 1)
                        if w_idx < w_idx_list[-1]:
                            w = self._create_spatial_weight(overlap_size//2).to(device)
                            out_patch[..., :, -overlap_size//2:] *= w.view(1, -1)
                            out_patch_mask[..., :, -overlap_size//2:] *= w.view(1, -1)
                        if h_idx > h_idx_list[0]:
                            w = self._create_spatial_weight(overlap_size//2).flip(0).to(device)
                            out_patch[..., :overlap_size//2, :] *= w.view(-1, 1)
                            out_patch_mask[..., :overlap_size//2, :] *= w.view(-1, 1)
                        if w_idx > w_idx_list[0]:
                            w = self._create_spatial_weight(overlap_size//2).flip(0).to(device)
                            out_patch[..., :, :overlap_size//2] *= w.view(1, -1)
                            out_patch_mask[..., :, :overlap_size//2] *= w.view(1, -1)
                    elif use_spatial_average:  # Cumulative average
                        # No processing needed, accumulate directly
                        pass
                    else:  # Hard transition
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

                    # Accumulate into the output
                    E[..., h_idx*sf:(h_idx+size_patch)*sf,
                      w_idx*sf:(w_idx+size_patch)*sf].add_(out_patch)
                    W[..., h_idx*sf:(h_idx+size_patch)*sf,
                      w_idx*sf:(w_idx+size_patch)*sf].add_(out_patch_mask)

            output = E.div_(W)

        else:
            # Process the full image
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

    def _create_temporal_weight(self, overlap_size):
        """Create temporal Gaussian weights."""
        x = torch.linspace(-3, 3, overlap_size)
        return torch.exp(-x**2 / 2)

    def _create_spatial_weight(self, overlap_size):
        """Create spatial Gaussian weights."""
        x = torch.linspace(-3, 3, overlap_size)
        return torch.exp(-x**2 / 2)

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
