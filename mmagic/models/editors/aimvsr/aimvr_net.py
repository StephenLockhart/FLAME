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
class AimVRNet(BaseModule):
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
                 feat_pretrained=None,
                 fre_decoder=True   # Whether to use the FreModule decoder
                 ):

        super().__init__()
        self.num_features = num_features
        self.scale_factor = scale_factor
        self.fre_decoder = fre_decoder

        if self.fre_decoder:
            # self.fre0 = FreModule(num_features, num_heads=4, bias=False)
            # self.fre1 = FreModule(num_features, num_heads=4, bias=False)
            self.fre2 = FreModule(num_features*2, num_heads=4, bias=False)
            # self.fre3 = FreModule(num_features, num_heads=4, bias=False)
            # Align channel number after residual concatenation
            self.conv_fre2_res = nn.Conv3d(num_features*2, num_features, kernel_size=(1, 3, 3), padding=(0, 1, 1)) # Align channels to 128

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

        if self.fre_decoder:
            M2 = rearrange(M2, 'b t c h w -> (b t) c h w')
            M2 = self.fre2(lqs_reshaped, M2)
            M2 = rearrange(M2, '(b t) c h w -> b t c h w', b=b)

        # 4. Decoder upsampling back to 64x64
        x_up = rearrange(M2, 'n d c h w -> n c d h w')
        # print(f"Before upsampling: {x_up.shape}")
        x_up = F.relu(self.upconv2(x_up))
        if self.fre_decoder:
            M1 = rearrange(M1, 'n d c h w -> n c d h w')
            # Ensure x_up and M1 match in height and width
            x_up = torch.cat([x_up, M1], 1) # Concatenate along the channel dimension
            x_up = self.conv_fre2_res(x_up) # Align channels to 128 after residual concatenation
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

                    
##########################################################################
## Adaptive Frequency Learning Block (AFLB)
class FreModule(nn.Module):
    def __init__(self, dim, num_heads, bias, in_dim=3):
        super(FreModule, self).__init__()

        self.conv = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Add normalization layers, using InstanceNorm which is more suitable for deraining and dehazing
        self.input_norm = nn.InstanceNorm2d(in_dim)
        self.feat_norm = nn.InstanceNorm2d(dim)
        
        self.score_gen = nn.Conv2d(2, 2, 7, padding=3)
        
        # Add constraints when initializing parameters
        self.para1 = nn.Parameter(torch.zeros(dim, 1, 1))   # Learnable scaling parameter
        self.para2 = nn.Parameter(torch.ones(dim, 1, 1))    # Learnable scaling parameter
        
        with torch.no_grad():
            self.para1.data.clamp_(-1, 1)
            self.para2.data.clamp_(0, 2)

        self.channel_cross_l = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)
        self.channel_cross_h = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)
        self.channel_cross_agg = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)

        self.frequency_refine = FreRefine(dim)

        # Modify rate_conv to use a GELU+Softplus combination
        self.rate_conv = nn.Sequential(
            nn.Conv2d(dim, dim//8, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim//8, 2, 1, bias=False),
            nn.Softplus()  # Use Softplus instead of Sigmoid, allowing a larger range while staying smooth
        )
        
        # Set appropriate numerical stability constants considering the input is normalized to [0,1]
        self.eps = 1e-5  # Slightly larger than float32 precision (1e-7) to avoid numerical instability; a value suitable for deraining and dehazing
        self.max_magnitude = 8.0  # A compromise between 10.0 and 5.0
        # self.min_freq_ratio = 0.1  # Minimum frequency block ratio
        # self.max_freq_ratio = 0.6  # Maximum frequency block ratio, increased to preserve more details

    def forward(self, x, y):
        # Remove the wrong monitoring position
        _, _, H, W = y.size()
        # Input normalization
        x = self.input_norm(x)
        x = F.interpolate(x, (H,W), mode='bilinear', align_corners=False)
        
        # Add pre-FFT monitoring
        if self.training and torch.rand(1) < 0.01:  # 1% probability sampling
            print(f"Pre-FFT input range: [{x.min():.3f}, {x.max():.3f}]")
        
        high_feature, low_feature = self.fft(x)
        
        # Add post-FFT monitoring
        if self.training and torch.rand(1) < 0.01:  # 1% probability sampling
            print(f"Post-FFT range: high[{high_feature.min():.3f}, {high_feature.max():.3f}], "
                  f"low[{low_feature.min():.3f}, {low_feature.max():.3f}]")
        
        # Feature normalization
        high_feature = self.feat_norm(high_feature)
        low_feature = self.feat_norm(low_feature)
        
        high_feature = self.channel_cross_l(high_feature, y)
        low_feature = self.channel_cross_h(low_feature, y)

        agg = self.frequency_refine(low_feature, high_feature)
        out = self.channel_cross_agg(y, agg)
        
        # Limit the output range
        para1 = torch.clamp(self.para1, -1, 1)
        para2 = torch.clamp(self.para2, 0, 2)
        
        return out * para1 + y * para2

    def shift(self, x):
        '''shift FFT feature map to center'''
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(int(h/2), int(w/2)), dims=(2,3))

    def unshift(self, x):
        """converse to shift operation"""
        b, c, h ,w = x.shape
        return torch.roll(x, shifts=(-int(h/2), -int(w/2)), dims=(2,3))

    def fft(self, x, n=128):
        """obtain high/low-frequency features from input"""            
        x = self.conv1(x)
        
        # Add input clamping
        x = torch.clamp(x, -self.max_magnitude, self.max_magnitude)
        
        # FFT monitoring
        if self.training and torch.rand(1) < 0.01:
            print(f"Pre-FFT conv range: [{x.min():.3f}, {x.max():.3f}]")
        
        # Mask generation
        mask = torch.zeros(x.shape, device=x.device)
        h, w = x.shape[-2:]
        
        # Compute adaptive thresholds; use exponential moving average to smooth threshold changes
        threshold = F.adaptive_avg_pool2d(x, 1)
        threshold = self.rate_conv(threshold)
        
        for i in range(mask.shape[0]):
            h_ = (h//n * threshold[i,0,:,:]).int()
            w_ = (w//n * threshold[i,1,:,:]).int()
            
            # Only add basic range constraints
            h_ = torch.clamp(h_, min=4, max=h//4)  # Minimum 4 pixels, maximum 1/4 of the image size
            w_ = torch.clamp(w_, min=4, max=w//4)
            
            mask[i, :, h//2-h_:h//2+h_, w//2-w_:w//2+w_] = 1
        
        # FFT computation
        try:
            x = x + self.eps
            # Ensure float32 dtype is used
            if x.dtype != torch.float32:
                x = x.float()
            fft = torch.fft.fft2(x, norm='forward')
            
            # FFT monitoring
            if self.training and torch.rand(1) < 0.01:
                print(f"FFT magnitude range: [{torch.abs(fft).min():.3f}, {torch.abs(fft).max():.3f}]")
            
            fft = self.shift(fft)

            # Separate high and low frequencies
            fft_high = fft * (1 - mask)
            fft_low = fft * mask
            
            high = self.unshift(fft_high)
            low = self.unshift(fft_low)
            
            high = torch.fft.ifft2(high, norm='forward')
            low = torch.fft.ifft2(low, norm='forward')
            
            # Use abs to get magnitudes, keeping a large but capped range
            high = torch.abs(high)
            low = torch.abs(low)
            
            # Clamp high and low frequency ranges separately
            high = torch.clamp(high, 0, self.max_magnitude)
            low = torch.clamp(low, 0, self.max_magnitude * 0.8)  # Clamp low-frequency magnitudes a bit more
            
        except RuntimeError as e:
            print(f"FFT computation failed: {e}")
            # Fallback when an error occurs
            high = x
            low = x
            
        # Cast the results back to the original dtype
        high = high.to(x.dtype) # Convert input to float32 for FFT computation, debugging
        low = low.to(x.dtype)   # Convert input to float32 for FFT computation, debugging

        return high, low

class Chanel_Cross_Attention(nn.Module):
    def __init__(self, dim, num_head, bias):
        super(Chanel_Cross_Attention, self).__init__()
        self.num_head = num_head
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1), requires_grad=True)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)


        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        # x -> q, y -> kv
        assert x.shape == y.shape, 'The shape of feature maps from image and features are not equal!'

        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_head)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_head, h=h, w=w)

        out = self.project_out(out)
        return out

##########################################################################
## Frequency Modulation Module (FMoM)
class FreRefine(nn.Module):
    def __init__(self, dim):
        super(FreRefine, self).__init__()

        self.SpatialGate = SpatialGate()
        self.ChannelGate = ChannelGate(dim)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, low, high):
        spatial_weight = self.SpatialGate(high)
        channel_weight = self.ChannelGate(low)
        high = high * channel_weight
        low = low * spatial_weight

        out = low + high
        out = self.proj(out)
        return out

##########################################################################
## H-L Unit
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()

        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        max = torch.max(x,1,keepdim=True)[0]
        mean = torch.mean(x,1,keepdim=True)
        scale = torch.cat((max, mean), dim=1)
        scale =self.spatial(scale)
        scale = F.sigmoid(scale)
        return scale

##########################################################################
## L-H Unit
class ChannelGate(nn.Module):
    def __init__(self, dim):
        super(ChannelGate, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.max = nn.AdaptiveMaxPool2d((1,1))

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim//16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(dim//16, dim, 1, bias=False)
        )

    def forward(self, x):
        avg = self.mlp(self.avg(x))
        max = self.mlp(self.max(x))

        scale = avg + max
        scale = F.sigmoid(scale)
        return scale