import torch
import torch.nn as nn
import torch.nn.functional as F
from mmagic.registry import MODELS
from torchvision.models import vgg19
from typing import Optional

@MODELS.register_module()
class ContrastLoss(nn.Module):
    """Contrast loss for comparing patches.

    Args:
        loss_weight (float): Loss weight for contrast loss. Default: 1.0.
        reduction (str): Reduction method. Choices are 'none', 'mean', 'sum'.
            Default: 'mean'.
        data_info (dict, optional): Dictionary contains the mapping between 
            loss input args and data dictionary. Default: None.
        ablation (bool): Whether to use ablation mode. Default: False.
    """
    def __init__(self,
                 loss_weight: float = 1.0,
                 reduction: str = 'mean',
                 data_info: Optional[dict] = None,
                 ablation: bool = False):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: none | mean | sum')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.data_info = data_info
        self.ablation = ablation
        
        # Initialize perceptual loss
        self.perceptual = PerceptualLoss2()

    def forward(self, *args, **kwargs):
        """Forward function.

        If data_info is not None, forward function will construct computational
        graph according to the data_info.

        Args:
            *args: Input arguments.
            **kwargs: Keyword arguments.

        Returns:
            Tensor: Loss tensor.
        """
        # Use data_info to build computational graph
        if self.data_info is not None:
            # Parse the args and kwargs
            if len(args) == 1:
                assert isinstance(args[0], dict)
                outputs_dict = args[0]
            elif 'outputs_dict' in kwargs:
                assert len(args) == 0
                outputs_dict = kwargs.pop('outputs_dict')
            else:
                raise NotImplementedError(
                    'Cannot parsing your arguments for contrast loss.')
                    
            # Build computational graph
            loss_input_dict = {
                k: outputs_dict[v]
                for k, v in self.data_info.items()
            }
            kwargs.update(loss_input_dict)
            return self._forward(**kwargs)
        else:
            return self._forward(*args, **kwargs)

    def _forward(self, 
                anchor: torch.Tensor,
                positive: torch.Tensor, 
                negative: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward function for contrast loss.

        Args:
            anchor (Tensor): Anchor image patches.
            positive (Tensor): Positive image patches.
            negative (Tensor): Negative image patches. 
            weight (Tensor, optional): Weight tensor. Default: None.

        Returns:
            Tensor: Loss tensor.
        """
        # Calculate distances
        d_ap = self.perceptual(anchor, positive.detach())
        d_an = self.perceptual(anchor, negative.detach())
        
        # Calculate contrast loss
        loss = d_ap / (d_an + 1e-7)

        # Apply weight if provided
        if weight is not None:
            loss = loss * weight

        # Apply reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss * self.loss_weight

    @staticmethod
    def loss_name() -> str:
        """Loss Name.

        Returns:
            str: Name of this loss item.
        """
        return 'loss_contrast'


class PerceptualLoss2(nn.Module):
    """VGG Perceptual loss.
    
    Args:
        pretrained (str): Path for pretrained VGG19 model. Default: True.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.criterion = nn.L1Loss()
        
        # Load pretrained VGG19 and move to same device
        vgg = vgg19(pretrained=pretrained).eval()
        device = next(vgg.parameters()).device
        vgg = vgg.to(device)
        
        # Extract specific layers
        self.layers = nn.ModuleList([
            nn.Sequential(*list(vgg.features)[:1]).eval(),
            nn.Sequential(*list(vgg.features)[:3]).eval()
        ])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (Tensor): Input tensor 1.
            y (Tensor): Input tensor 2.

        Returns:
            Tensor: Perceptual loss value.
        """
        # Calculate L1 loss for different layers
        losses = []
        for layer in self.layers:
            losses.append(self.criterion(layer(x), layer(y)))
            
        # Weighted combination of layer losses
        return sum(losses) / len(losses)