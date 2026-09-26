from mmagic.registry import MODELS
import torch
import torch.nn as nn
try:
    import lpips
except ImportError:
    raise ImportError('lpips package is not installed, please run pip install lpips')

@MODELS.register_module()
class LPIPSLoss(nn.Module):
    def __init__(self, loss_weight=1.0, net='alex', reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.lpips = lpips.LPIPS(net=net)
        self.lpips.eval()
        for p in self.lpips.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        # pred, target: [N,3,H,W], float, range [-1,1] or [0,1]
        loss = self.lpips(pred, target)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss * self.loss_weight 