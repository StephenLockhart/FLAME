import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

# from mmcv.runner import load_checkpoint
# from mmedit.models.registry import BACKBONES
# from mmedit.utils import get_root_logger
# MMagic 2.x uses the OpenMMLab 2.0 architecture
from mmengine import MMLogger  # New logging tool
from mmagic.registry import MODELS  # New model registry
from logging import WARNING

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine import MMLogger
from mmengine.model import BaseModule
from mmengine.runner import load_checkpoint

from mmagic.models.archs import PixelShufflePack, ResidualBlockNoBN
from mmagic.models.utils import flow_warp, make_layer

from mmengine.runner import load_checkpoint

from .modules.convnext import ConvNeXt
from .modules.head import ProjectionHead
from .modules.mambablock import MambaLayerglobal, MambaLayerlocal

@MODELS.register_module()
class AimNet(BaseModule):
    """AimNet network structure.

    Paper:
        AIM-VSR: All In Mamba for Video Super-Resolution

    Args:
        num_features (int, optional): Channel number of the intermediate
            features. Default: 128.
        scale_factor (int, optional): Scale factor for upsampling. Default: 4.

    """

    def __init__(self,
                 num_features=128,
                 scale_factor=4,
                 feat_pretrained=None):

        super().__init__()
        self.num_features = num_features
        self.scale_factor = scale_factor
        
        # Determine stem_patch_size based on whether super-resolution is used
        self.stem_patch_size = 1 if scale_factor > 1 else 4
        
        # ConvNeXt feature extraction
        self.feat_extract = ConvNeXt(
            arch='tiny',
            stem_patch_size=self.stem_patch_size,  # 1 for super-resolution, 4 for restoration
            out_indices=[0, 1, 2, 3],
            drop_path_rate=0.0,
            layer_scale_init_value=1.0,
            gap_before_final_norm=False,
            init_cfg=dict(type='Pretrained', checkpoint=feat_pretrained, prefix='backbone.'))

        self.head = ProjectionHead(in_channels=[96, 192, 384, 768],
                                   out_channels=num_features,
                                   num_outs=4
                                   )

        # self.backbone = nn.ModuleDict()

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

        self.refine = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1)
        )

        # downsample
        self.conv1 = nn.Conv3d(num_features, num_features*2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))

        # upsample
        self.upconv2 = nn.ConvTranspose3d(num_features*2, num_features, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1),
                                          output_padding=(0, 1, 1))
        self.conv_before_upsample1 = nn.Conv3d(num_features, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1)) # Align channels to 128
        self.upsample1 = nn.PixelShuffle(2)
        self.conv_before_upsample2 = nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))    # Unclear why channels go from 32 back up to 64; consider keeping channels unchanged or decreasing
        self.upsample2 = nn.PixelShuffle(2)
        self.conv_last = nn.Conv3d(16, 3, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # 64 * 64 * 128
        self.GlobalMambaBlock1 = MambaLayerglobal(dim=num_features)
        self.LocalMambaBlock1 = MambaLayerlocal(dim=num_features)
        self.GlobalMambaBlock2 = MambaLayerglobal(dim=num_features)
        self.LocalMambaBlock2 = MambaLayerlocal(dim=num_features)

        # 32 * 32 * 256
        self.GlobalMambaBlockLowRes1 = MambaLayerglobal(dim=num_features * 2)
        self.LocalMambaBlockLowRes1 = MambaLayerlocal(dim=num_features * 2)
        self.GlobalMambaBlockLowRes2 = MambaLayerglobal(dim=num_features * 2)
        self.LocalMambaBlockLowRes2 = MambaLayerlocal(dim=num_features * 2)
        self.GlobalMambaBlockLowRes3 = MambaLayerglobal(dim=num_features * 2)
        self.LocalMambaBlockLowRes3 = MambaLayerlocal(dim=num_features * 2)

        # 64 * 64 * 128
        self.GlobalMambaBlock3 = MambaLayerglobal(dim=num_features)
        self.LocalMambaBlock3 = MambaLayerlocal(dim=num_features)
        self.GlobalMambaBlock4 = MambaLayerglobal(dim=num_features)
        self.LocalMambaBlock4 = MambaLayerlocal(dim=num_features)

    def check_if_mirror_extended(self, lqs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
        """

        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def forward(self, lqs, hilbert_curve_large_scale, hilbert_curve_small_scale):
        """Forward function.
        
        Args:
            lqs (Tensor): Input tensor with shape [B, T, C, H, W]
                - When scale_factor=1, input and output have the same size (deraining mode)
                - When scale_factor=4, the input is a downsampled small image (super-resolution mode)
            hilbert_curve_large_scale (Tensor): Hilbert curve for the 64x64 feature map
            hilbert_curve_small_scale (Tensor): Hilbert curve for the 32x32 feature map
            
        Returns:
            Tensor: Output tensor with shape [B, T, C, H*scale, W*scale]
        """
        # Get input dimensions [B, T, C, H, W]
        b, t, c, h, w = lqs.size()
        # print(f"Input tensor shape: b={b}, t={t}, c={c}, h={h}, w={w}")
        # print(f"Model mode: {'Super-resolution' if self.scale_factor > 1 else 'Restoration'}")
        # print(f"Scale factor: {self.scale_factor}, Stem patch size: {self.stem_patch_size}")
        
        # Check whether the frame count matches the expectation (5 frames)
        # expected_frames = 5
        # if t != expected_frames:
        #   raise ValueError(f"Input frames ({t}) do not match expected frames ({expected_frames})")
        
        # 2. ConvNeXt feature extraction - adjust dimension order to [B*T, C, H, W]
        lqs_reshaped = lqs.view(-1, c, h, w)
        # print(f"Reshaped input: {lqs_reshaped.shape}")
        
        # ConvNeXt feature extraction - returns a tuple of multi-scale features
        feats = self.feat_extract(lqs_reshaped)
        # print(f"Feature extraction outputs: {[f.shape for f in feats]}")
        
        # Process multi-scale features with ProjectionHead
        down1, down2, down3, down4 = self.head(feats)
        # print(f"Feature maps shapes: {down1.shape}, {down2.shape}, {down3.shape}, {down4.shape}")
        
        # 3. Feature fusion - upsample and fuse multi-scale features
        down2_up = F.interpolate(down2, size=down1.size()[2:], mode='bilinear', align_corners=True)
        down3_up = F.interpolate(down3, size=down1.size()[2:], mode='bilinear', align_corners=True)
        down4_up = F.interpolate(down4, size=down1.size()[2:], mode='bilinear', align_corners=True)
        f = (down1 + down2_up + down3_up + down4_up) / 4
        f = self.refine(f) + f
        f_ori = f
        
        # Correctly compute the feature map size
        _, c, feat_h, feat_w = f_ori.shape
        # print(f"f_ori.shape, computed feature map size: {f_ori.shape}")
        # Reshape into temporal features [B, T, C, H, W]
        x_new = f_ori.view(b, t, self.num_features, feat_h, feat_w)
        # print(f"Reshaped features before Mamba: {x_new.shape}")
        
        # Mamba processing - high-resolution path (64x64)
        M1 = self.GlobalMambaBlock1(x_new)
        # print(f"After GlobalMambaBlock1: {M1.shape}")
        M1 = self.LocalMambaBlock1(M1, hilbert_curve_large_scale)
        # print(f"After LocalMambaBlock1: {M1.shape}")
        M1 = self.GlobalMambaBlock2(M1)
        # print(f"After GlobalMambaBlock2: {M1.shape}")
        M1 = self.LocalMambaBlock2(M1, hilbert_curve_large_scale)
        # print(f"After LocalMambaBlock2: {M1.shape}")

        # Downsample to 32x32
        x_down = rearrange(M1, 'n d c h w -> n c d h w')
        # print(f"Before downsampling: {x_down.shape}")
        x_down = F.relu(self.conv1(x_down))
        x_down = rearrange(x_down, 'n c d h w -> n d c h w')
        # print(f"After downsampling: {x_down.shape}")

        # Mamba processing - low-resolution path (32x32)
        M2 = self.GlobalMambaBlockLowRes1(x_down)
        M2 = self.LocalMambaBlockLowRes1(M2, hilbert_curve_small_scale)
        M2 = self.GlobalMambaBlockLowRes2(M2)
        M2 = self.LocalMambaBlockLowRes2(M2, hilbert_curve_small_scale)
        M2 = self.GlobalMambaBlockLowRes3(M2)
        M2 = self.LocalMambaBlockLowRes3(M2, hilbert_curve_small_scale)
        
        # 4. Decoder upsampling back to 64x64
        x_up = rearrange(M2, 'n d c h w -> n c d h w')
        # print(f"Before upsampling: {x_up.shape}")
        x_up = F.relu(self.upconv2(x_up))
        x_up = rearrange(x_up, 'n c d h w -> n d c h w')
        # print(f"After upsampling: {x_up.shape}")

        # Final Mamba processing
        M3 = self.GlobalMambaBlock3(x_up)
        M3 = self.LocalMambaBlock3(M3, hilbert_curve_large_scale)
        M3 = self.GlobalMambaBlock4(M3)
        M3 = self.LocalMambaBlock4(M3, hilbert_curve_large_scale)

        # Reconstruct the output
        x_re = rearrange(M3, 'n d c h w -> n c d h w')
        x_re = self.up(x_re)
        final = self.conv_last(x_re)
        
        # Adjust dimension order to [B, T, C, H, W]
        final = final.permute(0, 2, 1, 3, 4)  # -> [B, T, C, H, W]
        # print(f"Final output shape: {final.shape}")
        
        return final


    def up(self, x):
        x = self.conv_before_upsample1(x)  # [B, 128, T, H, W]
        x = rearrange(x, 'n c d h w -> n d c h w')
        x = self.upsample1(x)  # Channels become 32 (128/4)
        x = rearrange(x, 'n d c h w -> n c d h w')
        
        x = self.conv_before_upsample2(x)  # [B, 64, T, H, W]
        x = rearrange(x, 'n c d h w -> n d c h w')
        x = self.upsample2(x)  # Channels become 16 (64/4)
        x = rearrange(x, 'n d c h w -> n c d h w')
        
        return x  # Ensure the final channel number is correct

    def init_weights(self, pretrained=None, strict=False):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
            strict (bool, optional): Whether strictly load the pretrained model.
                Default: True.
        """
        # 1. Load the pretrained model
        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            logger.info(f'Load model from: {pretrained}')
            # Ensure the pretrained model path is a string
            if not isinstance(pretrained, (str, bytes)):
                raise TypeError(f'pretrained must be a str or bytes, but got {type(pretrained)}')
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        
        # 2. Initialize the feature extractor
        elif hasattr(self.feat_extract, 'init_cfg') and self.feat_extract.init_cfg is not None:
            if isinstance(self.feat_extract.init_cfg.get('checkpoint'), (str, bytes)):
                self.feat_extract.init_weights()
            else:
                logger = MMLogger.get_current_instance()
                logger.warning('Skip initializing feat_extract due to invalid checkpoint path')
                self._init_other_modules()