# Copyright (c) OpenMMLab. All rights reserved.
import torch
from typing import Optional
import warnings

from mmagic.registry import METRICS
from .base_sample_wise_metric import BaseSampleWiseMetric
from mmagic.utils import to_numpy

# Check lpips availability at module level
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    warnings.warn(
        "LPIPS package not found. Please install with: pip install lpips. "
        "LPIPS metric will be disabled.",
        UserWarning
    )


@METRICS.register_module()
class LPIPS(BaseSampleWiseMetric):
    """Learned Perceptual Image Patch Similarity (LPIPS) metric.
    
    Args:
        net (str): Network type for LPIPS computation. Options: 'alex', 'vgg', 'squeeze'. Default: 'alex'.
        gt_key (str): Key of ground-truth. Default: 'gt_img'
        pred_key (str): Key of prediction. Default: 'pred_img'
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None

    Metrics:
        - LPIPS (float): Learned Perceptual Image Patch Similarity
    """
    
    metric = 'LPIPS'

    def __init__(self, 
                 net: str = 'alex',
                 gt_key: str = 'gt_img',
                 pred_key: str = 'pred_img',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(
            gt_key=gt_key,
            pred_key=pred_key,
            mask_key=None,
            collect_device=collect_device,
            prefix=prefix)
        
        if not LPIPS_AVAILABLE:
            raise ImportError(
                'LPIPS package not found. Please install with: pip install lpips'
            )
            
        self.net = net
        self.lpips_loss = lpips.LPIPS(net=net)
        
        # Move to device if CUDA is available
        if torch.cuda.is_available():
            self.lpips_loss = self.lpips_loss.cuda()

    def process_image(self, gt, pred, mask):
        """Process an image.

        Args:
            gt (Torch | np.ndarray): GT image.
            pred (Torch | np.ndarray): Pred image.
            mask (Torch | np.ndarray): Mask of evaluation.
        Returns:
            float: LPIPS result.
        """
        # Convert to numpy arrays first
        gt = to_numpy(gt)
        pred = to_numpy(pred)
        
        # Convert to torch tensors
        gt_tensor = torch.from_numpy(gt).float()
        pred_tensor = torch.from_numpy(pred).float()
        
        # Ensure correct input order (CHW)
        if gt_tensor.dim() == 3 and gt_tensor.shape[0] != 3:
            gt_tensor = gt_tensor.permute(2, 0, 1)  # HWC -> CHW
            pred_tensor = pred_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        # Normalize to [0, 1] if needed, then to [-1, 1]
        if gt_tensor.max() > 1.0:
            gt_tensor = gt_tensor / 255.0
            pred_tensor = pred_tensor / 255.0
        
        gt_tensor = gt_tensor * 2.0 - 1.0
        pred_tensor = pred_tensor * 2.0 - 1.0
        
        # Add batch dimension
        gt_tensor = gt_tensor.unsqueeze(0)
        pred_tensor = pred_tensor.unsqueeze(0)
        
        # Move to same device as LPIPS model
        if torch.cuda.is_available():
            gt_tensor = gt_tensor.cuda()
            pred_tensor = pred_tensor.cuda()
        
        # Compute LPIPS
        with torch.no_grad():
            lpips_score = self.lpips_loss(pred_tensor, gt_tensor)
        
        return lpips_score.cpu().item() 