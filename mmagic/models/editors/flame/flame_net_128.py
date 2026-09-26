import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import importlib.util
import sys
import os
import warnings
from einops import rearrange
from einops.layers.torch import Rearrange

# Filter torch.meshgrid indexing warning to avoid modifying external VRT library
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument")

from mmengine import MMLogger
from mmengine.model import BaseModule
from mmengine.runner import load_checkpoint
from mmagic.registry import MODELS

# Reuse AimVR modules
from ..aimvsr.aimvr_net import AimVRNet

# Reuse VRT modules
def load_vrt_modules():
    current_dir = os.path.dirname(__file__)
    vrt_path = os.path.join(current_dir, 'VRT-main', 'models', 'network_vrt.py')

    spec = importlib.util.spec_from_file_location("network_vrt", vrt_path)
    network_vrt = importlib.util.module_from_spec(spec)
    sys.modules["network_vrt"] = network_vrt
    spec.loader.exec_module(network_vrt)

    return network_vrt.Stage, network_vrt.SpyNet, network_vrt.DCNv2PackFlowGuided, network_vrt.Mlp_GEGLU

Stage, SpyNet, DCNv2PackFlowGuided, Mlp_GEGLU = load_vrt_modules()


class FlowGuidedFusion128(nn.Module):
    """128-patch-specific VRT optical flow fusion module - fully reuses logic from flame_net, only adjusts feature map sizes"""

    def __init__(self, mamba_dim, vrt_dim=96, pa_frames=6, deformable_groups=16,
                 trsa_depth=2, trsa_heads=6, window_size=[6, 8, 8]):
        super().__init__()
        self.mamba_dim = mamba_dim  # 256 or 512
        self.vrt_dim = vrt_dim      # 96
        self.requested_pa_frames = pa_frames  # 0/2/4/6, user-requested pa_frames

        # Step 1: Dimension reduction (before VRT)
        self.feat_to_vrt = nn.Conv3d(mamba_dim, vrt_dim, 1)

        # Step 2: Smart VRT Stage - 128-patch version, feature map size adjusted to 32x32
        print(f"FlowGuidedFusion128: mamba_dim={mamba_dim}, vrt_dim={vrt_dim}, "
              f"input_resolution=(dynamic,32,32), pa_frames={pa_frames}")

        # No-flow VRT Stage (pa_frames=0) - 128-patch: 32x32 feature map
        self.vrt_stage_no_flow = Stage(
            in_dim=vrt_dim,
            dim=vrt_dim,
            input_resolution=(window_size[0], 32, 32),  # 128-patch corresponds to 32x32 feature map
            depth=trsa_depth,
            num_heads=trsa_heads,
            window_size=window_size,
            pa_frames=0,  # Force no-flow mode
            deformable_groups=deformable_groups,
            reshape='none',
            mlp_ratio=2.0,
            qkv_bias=True,
            use_checkpoint_attn=False,
            use_checkpoint_ffn=False,
        )

        # Flow VRT Stage (actual pa_frames) - 128-patch: 32x32 feature map
        if pa_frames > 0:
            self.vrt_stage_with_flow = Stage(
                in_dim=vrt_dim,
                dim=vrt_dim,
                input_resolution=(window_size[0], 32, 32),  # 128-patch corresponds to 32x32 feature map
                depth=trsa_depth,
                num_heads=trsa_heads,
                window_size=window_size,
                pa_frames=pa_frames,  # Use actual pa_frames value
                deformable_groups=deformable_groups,  # Use adjusted value
                reshape='none',
                mlp_ratio=2.0,
                qkv_bias=True,
                use_checkpoint_attn=False,
                use_checkpoint_ffn=False,
            )
        else:
            self.vrt_stage_with_flow = None

        # Step 3: Feature recovery (after VRT processing)
        self.vrt_to_feat = nn.Conv3d(vrt_dim, vrt_dim, 1)

    def _update_vrt_input_resolution(self, vrt_stage, actual_shape):
        """Dynamically update VRT Stage's input_resolution - 128-patch version"""
        t, h, w = actual_shape
        new_resolution = (t, h, w)

        # Update TMSAG module's input_resolution
        if hasattr(vrt_stage, 'residual_group1'):
            vrt_stage.residual_group1.input_resolution = new_resolution
            # Update TMSA blocks inside TMSAG
            for block in vrt_stage.residual_group1.blocks:
                if hasattr(block, 'input_resolution'):
                    block.input_resolution = new_resolution

        if hasattr(vrt_stage, 'residual_group2'):
            vrt_stage.residual_group2.input_resolution = new_resolution
            # Update TMSA blocks inside TMSAG
            for block in vrt_stage.residual_group2.blocks:
                if hasattr(block, 'input_resolution'):
                    block.input_resolution = new_resolution

    def forward(self, mamba_features, flows_backward=None, flows_forward=None):
        """128-patch-specific forward pass - fully reuses logic from flame_net"""
        try:
            # Step 1: Reduce Mamba feature dimensions to VRT dimension
            mamba_3d = rearrange(mamba_features, 'b t c h w -> b c t h w')
            vrt_3d = self.feat_to_vrt(mamba_3d)  # [B, vrt_dim, T, H, W]

            # Dynamically update input_resolution
            b, c, t, h, w = vrt_3d.shape
            actual_shape = (t, h, w)

            # Step 2: Smart VRT Stage selection
            has_valid_flows = (flows_backward is not None and flows_forward is not None and
                              flows_backward.numel() > 0 and flows_forward.numel() > 0)

            if self.requested_pa_frames > 0 and has_valid_flows and self.vrt_stage_with_flow is not None:
                # Fix: check optical flow format expected by VRT Stage
                try:
                    b_flow, t_minus_1, c_flow, h_flow, w_flow = flows_backward.shape  # [B, T-1, 2, H, W]
                    # Step 1: 1-frame-interval optical flow (use directly)
                    flows_backward_1 = flows_backward  # [B, T-1, 2, H, W]
                    flows_forward_1 = flows_forward

                    # Fix: uniformly use 3 optical flows, consistent with original flame_net.py
                    flows_backward_2 = self._generate_2frame_flows(flows_backward_1, flows_forward_1, backward=True)
                    flows_forward_2 = self._generate_2frame_flows(flows_backward_1, flows_forward_1, backward=False)
                    flows_backward_3 = self._generate_3frame_flows(flows_backward_1, flows_forward_1, flows_backward_2, flows_forward_2, backward=True)
                    flows_forward_3 = self._generate_3frame_flows(flows_backward_1, flows_forward_1, flows_backward_2, flows_forward_2, backward=False)

                    # Build optical flow list format expected by VRT Stage
                    flows_backward_list = [flows_backward_1, flows_backward_2, flows_backward_3]
                    flows_forward_list = [flows_forward_1, flows_forward_2, flows_forward_3]

                    self._update_vrt_input_resolution(self.vrt_stage_with_flow, actual_shape)
                    vrt_enhanced = self.vrt_stage_with_flow(vrt_3d, flows_backward_list, flows_forward_list)

                except Exception as e:
                    print(f"VRT Stage with flow processing failed: {e}, switching to no-flow mode")
                    # Fallback to no-flow mode
                    self._update_vrt_input_resolution(self.vrt_stage_no_flow, actual_shape)
                    vrt_enhanced = self.vrt_stage_no_flow(vrt_3d, None, None)
            else:
                # Use no-flow VRT Stage
                self._update_vrt_input_resolution(self.vrt_stage_no_flow, actual_shape)
                # No-flow mode does not need optical flow parameters
                vrt_enhanced = self.vrt_stage_no_flow(vrt_3d, None, None)

            # Step 3: Feature refinement
            vrt_output = self.vrt_to_feat(vrt_enhanced)  # [B, vrt_dim, T, H, W]

            # Convert back to [B, T, vrt_dim, H, W]
            vrt_features = rearrange(vrt_output, 'b c t h w -> b t c h w')

            return vrt_features

        except Exception as e:
            print(f"FlowGuidedFusion128 processing failed: {e}, returning dimension-reduced Mamba features")
            # On complete failure, return dimension-reduced Mamba features
            mamba_3d = rearrange(mamba_features, 'b t c h w -> b c t h w')
            # Fix: correctly handle feature dimension reduction
            if mamba_3d.shape[1] != self.vrt_dim:
                # Use 1x1x1 convolution for channel dimension conversion
                fallback_conv = F.conv3d(mamba_3d,
                                       weight=torch.randn(self.vrt_dim, mamba_3d.shape[1], 1, 1, 1,
                                                         device=mamba_3d.device, dtype=mamba_3d.dtype) * 0.02,
                                       bias=None)
                fallback_features = fallback_conv
            else:
                fallback_features = mamba_3d
            return rearrange(fallback_features, 'b c t h w -> b t c h w')

    @staticmethod
    def _generate_2frame_flows(flows_backward_1, flows_forward_1, backward=True):
        """Generate 2-frame-interval optical flow based on VRT get_flow_4frames algorithm - fully reuses VRT algorithm"""
        from mmagic.models.utils import flow_warp

        b, t_minus_1, c, h, w = flows_backward_1.shape
        t = t_minus_1 + 1  # Actual number of frames

        if backward:
            # VRT algorithm: backward 2-frame-interval optical flow computation
            flow_list = []
            for i in range(t_minus_1 - 1, 0, -1):  # Start from second-to-last frame in reverse order
                flow_n1 = flows_backward_1[:, i - 1, :, :, :]  # flow from i+1 to i
                flow_n2 = flows_backward_1[:, i, :, :, :]      # flow from i+2 to i+1
                # True optical flow accumulation: flow from i+2 to i = flow_n1 + warp(flow_n2, flow_n1)
                flow_combined = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                flow_list.insert(0, flow_combined)

            if flow_list:
                flows_2 = torch.stack(flow_list, 1)  # [B, T-2, 2, H, W]
            else:
                # Create zero tensor when insufficient
                flows_2 = torch.zeros(b, max(1, t_minus_1-1), c, h, w,
                                    device=flows_backward_1.device, dtype=flows_backward_1.dtype)
        else:
            # VRT algorithm: forward 2-frame-interval optical flow computation
            flow_list = []
            for i in range(1, t_minus_1):  # Start from frame 2
                flow_n1 = flows_forward_1[:, i, :, :, :]      # flow from i-1 to i
                flow_n2 = flows_forward_1[:, i - 1, :, :, :] # flow from i-2 to i-1
                # True optical flow accumulation: flow from i-2 to i = flow_n1 + warp(flow_n2, flow_n1)
                flow_combined = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                flow_list.append(flow_combined)

            if flow_list:
                flows_2 = torch.stack(flow_list, 1)  # [B, T-2, 2, H, W]
            else:
                # Create zero tensor when insufficient
                flows_2 = torch.zeros(b, max(1, t_minus_1-1), c, h, w,
                                    device=flows_forward_1.device, dtype=flows_forward_1.dtype)

        return flows_2

    @staticmethod
    def _generate_3frame_flows(flows_backward_1, flows_forward_1,
                               flows_backward_2, flows_forward_2, backward=True):
        """Generate 3-frame-interval optical flow based on VRT get_flow_6frames algorithm - fully reuses VRT algorithm"""
        from mmagic.models.utils import flow_warp

        if backward:
            # VRT algorithm: backward 3-frame-interval optical flow computation
            t_minus_2 = flows_backward_2.shape[1]  # T-2
            flow_list = []
            for i in range(t_minus_2 - 1, 0, -1):  # Start from second-to-last frame in reverse order
                flow_n1 = flows_backward_2[:, i - 1, :, :, :]  # flow from i+2 to i
                flow_n2 = flows_backward_1[:, i + 1, :, :, :]  # flow from i+3 to i+2
                # True optical flow accumulation: flow from i+3 to i = flow_n1 + warp(flow_n2, flow_n1)
                flow_combined = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                flow_list.insert(0, flow_combined)

            if flow_list:
                flows_3 = torch.stack(flow_list, 1)  # [B, T-3, 2, H, W]
            else:
                # Create zero tensor
                b, _, c, h, w = flows_backward_2.shape
                flows_3 = torch.zeros(b, max(1, t_minus_2-1), c, h, w,
                                    device=flows_backward_2.device, dtype=flows_backward_2.dtype)
        else:
            # VRT algorithm: forward 3-frame-interval optical flow computation
            t_minus_2 = flows_forward_2.shape[1]  # T-2
            flow_list = []
            for i in range(2, t_minus_2 + 1):  # Start from frame 3
                flow_n1 = flows_forward_2[:, i - 1, :, :, :]  # flow from i-2 to i
                flow_n2 = flows_forward_1[:, i - 2, :, :, :]  # flow from i-3 to i-2
                # True optical flow accumulation: flow from i-3 to i = flow_n1 + warp(flow_n2, flow_n1)
                flow_combined = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                flow_list.append(flow_combined)

            if flow_list:
                flows_3 = torch.stack(flow_list, 1)  # [B, T-3, 2, H, W]
            else:
                # Create zero tensor
                b, _, c, h, w = flows_forward_2.shape
                flows_3 = torch.zeros(b, max(1, t_minus_2-1), c, h, w,
                                    device=flows_forward_2.device, dtype=flows_forward_2.dtype)

        return flows_3

    def _compute_flow_mask_weather_robust(self, flows_backward, flows_forward):
        """Compute weather-robust optical flow quality mask - specifically designed for weather degradation and occlusion"""
        b, t_minus_1, _, h, w = flows_backward.size()
        t = t_minus_1 + 1

        # 1. Optical flow magnitude analysis (weather degradation detection)
        mag_backward = torch.norm(flows_backward, dim=2, keepdim=True)  # [B, T-1, 1, H, W]
        mag_forward = torch.norm(flows_forward, dim=2, keepdim=True)
        mag_avg = (mag_backward + mag_forward) / 2

        # Weather degradation feature: abnormally large optical flow values usually indicate degradation
        mag_median = torch.median(mag_avg.reshape(b, t_minus_1, -1), dim=2, keepdim=True)[0]  # [B, T-1, 1]
        mag_median = mag_median.unsqueeze(-1).unsqueeze(-1)  # [B, T-1, 1, 1, 1]

        # Abnormal flow detection: regions exceeding 3x median may be weather degradation
        abnormal_flow = (mag_avg > 3 * mag_median).float()
        weather_confidence = 1.0 - abnormal_flow  # Low confidence in abnormal regions

        # 2. Forward-backward consistency (occlusion detection)
        consistency_error = torch.norm(
            flows_backward + flows_forward,  # Should be close to 0 ideally
            dim=2, keepdim=True
        )  # [B, T-1, 1, H, W]

        # Adaptive consistency threshold
        consistency_median = torch.median(
            consistency_error.reshape(b, t_minus_1, -1), dim=2, keepdim=True
        )[0].unsqueeze(-1).unsqueeze(-1)  # [B, T-1, 1, 1, 1]

        # Occlusion detection: regions where consistency error exceeds 2x median may be occluded
        occlusion_mask = (consistency_error > 2 * consistency_median).float()
        occlusion_confidence = 1.0 - occlusion_mask  # Low confidence in occluded regions

        # 3. Optical flow smoothness (texture region detection)
        flow_grad_x = torch.abs(flows_backward[:, :, :, :, 1:] - flows_backward[:, :, :, :, :-1])
        flow_grad_y = torch.abs(flows_backward[:, :, :, 1:, :] - flows_backward[:, :, :, :-1, :])

        # Fix: use constant mode to avoid replicate mode issues on 5D tensors
        flow_grad_x = F.pad(flow_grad_x, (0, 1, 0, 0), mode='constant', value=0)
        flow_grad_y = F.pad(flow_grad_y, (0, 0, 0, 1), mode='constant', value=0)

        flow_smoothness = torch.norm(torch.cat([flow_grad_x, flow_grad_y], dim=2), dim=2, keepdim=True)
        smoothness_median = torch.median(
            flow_smoothness.reshape(b, t_minus_1, -1), dim=2, keepdim=True
        )[0].unsqueeze(-1).unsqueeze(-1)

        # Smooth regions are more reliable
        smooth_confidence = torch.sigmoid(-(flow_smoothness - smoothness_median))

        # 4. Comprehensive quality score (simple weighted combination)
        quality_score = (0.4 * weather_confidence +
                        0.35 * occlusion_confidence +
                        0.25 * smooth_confidence)  # [B, T-1, 1, H, W]

        # Set reasonable confidence range [0.2, 1.0] to avoid complete suppression
        quality_score = torch.clamp(quality_score, 0.2, 1.0)

        # 5. Extend to full temporal dimension [B, T, 1, H, W]
        first_frame_score = quality_score[:, 0:1, :, :, :]  # [B, 1, 1, H, W]
        flow_mask = torch.cat([first_frame_score, quality_score], dim=1)  # [B, T, 1, H, W]

        return flow_mask


# Reuse AdaptiveFusion from flame_net
from .flame_net import AdaptiveFusion


@MODELS.register_module(name=['FlameNet128', 'DMVRTNetFixed128'])
class FlameNet128(BaseModule):
    """128-patch-specific version of FlameNet

    Specifically optimized for 128x128 patch training, fully reuses logic from flame_net.py:
    - Main difference: feature map sizes in optical flow computation (32x32 vs original 64x64, 16x16 vs original 32x32)
    - All other logic is identical to ensure stability
    """

    def __init__(self,
                 num_features=256,
                 vrt_dim=96,  # VRT processing dimension, aligned with original VRT
                 scale_factor=1,
                 img_size=[6, 32, 32],  # Feature map size for 128-patch
                 window_size=[6, 8, 8],
                 pa_frames=6,  # Use all 6 frames
                 feat_pretrained=None,
                 spynet_path=None,
                 trsa64a=True,
                 trsa32=False,
                 trsa64b=True,
                 fre_decoder=True,
                 use_flow_mask=True,
                 deformable_groups=16,  # For vrt_dim=96, use 16 (consistent with original VRT: 96/16=6)
                 trsa64_heads=6,  # Consistent with VRT
                 trsa64_depth=2,
                 trsa32_heads=6,  # Consistent with VRT
                 trsa32_depth=2,
                 **kwargs):

        super().__init__()
        self.num_features = num_features
        self.vrt_dim = vrt_dim
        self.scale_factor = scale_factor
        self.pa_frames = pa_frames
        self.requested_pa_frames = pa_frames
        self.img_size = img_size
        self.fre_decoder = fre_decoder

        # TRSA switches
        self.trsa64a = trsa64a
        self.trsa32 = trsa32
        self.trsa64b = trsa64b

        # Switch configuration: only keep flow_mask; optical flow logic controlled by pa_frames
        self.use_flow_mask = use_flow_mask

        # 1. Fully reuse AIM-VR components
        self._init_aimvr_components(feat_pretrained)

        # 2. Optical flow estimator (reuse VRT's SpyNet)
        self.spynet = None
        if self.pa_frames > 0 and spynet_path is not None:
            try:
                if isinstance(spynet_path, str) and len(spynet_path.strip()) > 0:
                    self.spynet = SpyNet(spynet_path, [2, 3, 4, 5])
                    print(f"SpyNet initialized successfully, pa_frames={self.pa_frames}")
                else:
                    raise ValueError("spynet_path is empty or invalid")
            except Exception as e:
                print(f"SpyNet initialization failed: {e}")
                print("Forcing fallback to no-flow mode...")
                self.pa_frames = 0
                self.spynet = None
        else:
            print(f"Skipping SpyNet initialization: pa_frames={self.pa_frames}, spynet_path={spynet_path}")
            if self.pa_frames <= 0:
                self.pa_frames = 0
            self.spynet = None

        # 3. FlowGuidedFusion128 module (64A position: after M1)
        if self.trsa64a:
            self.dmvrt_fusion_64a = FlowGuidedFusion128(
                mamba_dim=num_features,  # 256
                vrt_dim=vrt_dim,         # 96
                pa_frames=self.pa_frames,
                deformable_groups=deformable_groups,  # 16
                trsa_depth=trsa64_depth, # 2
                trsa_heads=trsa64_heads, # 6
                window_size=window_size
            )
            self.advanced_fusion_64a = AdaptiveFusion(
                mamba_dim=num_features,  # 256
                vrt_dim=vrt_dim,         # 96
                use_flow_mask=use_flow_mask,
                flow_mask_strength=0.5,
                scale_info="deepA"
            )

        # 4. FlowGuidedFusion128 module (32 position: after M2, before Fre)
        if self.trsa32:
            self.dmvrt_fusion_32 = FlowGuidedFusion128(
                mamba_dim=num_features * 2,  # 512
                vrt_dim=vrt_dim,             # 96
                pa_frames=self.pa_frames,
                deformable_groups=deformable_groups,  # 16
                trsa_depth=trsa32_depth,     # 2
                trsa_heads=trsa32_heads,     # 6
                window_size=window_size
            )
            self.advanced_fusion_32 = AdaptiveFusion(
                mamba_dim=num_features * 2,  # 512
                vrt_dim=vrt_dim,             # 96
                use_flow_mask=use_flow_mask,
                scale_info="latent"
            )

        # 5. FlowGuidedFusion128 module (64B position: after M3)
        if self.trsa64b:
            self.dmvrt_fusion_64b = FlowGuidedFusion128(
                mamba_dim=num_features,  # 256
                vrt_dim=vrt_dim,         # 96
                pa_frames=self.pa_frames,
                deformable_groups=deformable_groups,  # 16
                trsa_depth=trsa64_depth, # 2
                trsa_heads=trsa64_heads, # 6
                window_size=window_size
            )
            self.advanced_fusion_64b = AdaptiveFusion(
                mamba_dim=num_features,  # 256
                vrt_dim=vrt_dim,         # 96
                use_flow_mask=use_flow_mask,
                scale_info="deepB"
            )

        # DDP fix: completely disable all checkpoint functionality
        self._fix_ddp_checkpoint_issue()

    def _init_aimvr_components(self, feat_pretrained):
        """Fully reuse all AIM-VR components - identical to flame_net.py"""
        from ..aimvsr.modules.convnext import ConvNeXt
        from ..aimvsr.modules.head import ProjectionHead
        from ..aimvsr.modules.mambablock import MambaLayerglobal, MambaLayerlocal

        # Frequency enhancement module
        if self.fre_decoder:
            from ..aimvsr.aimvr_net import FreModule
            self.fre2 = FreModule(self.num_features*2, num_heads=4, bias=False)
            # Channel alignment after residual concat - unified 2.5D convolution fusion
            self.conv_fre2_res = nn.Conv3d(self.num_features*2, self.num_features,
                                         kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # Determine stem_patch_size based on whether super-resolution is used
        stem_patch_size = 1 if self.scale_factor > 1 else 4

        # ConvNeXt feature extraction
        self.feat_extract = ConvNeXt(
            arch='tiny',
            stem_patch_size=stem_patch_size,
            out_indices=[0, 1, 2, 3],
            drop_path_rate=0.0,
            layer_scale_init_value=1.0,
            gap_before_final_norm=False,
            init_cfg=dict(type='Pretrained', checkpoint=feat_pretrained, prefix='backbone.')
            if feat_pretrained else None
        )

        self.head = ProjectionHead(
            in_channels=[96, 192, 384, 768],
            out_channels=self.num_features,
            num_outs=4
        )

        self.refine = nn.Sequential(
            nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=1),
            nn.SELU(),
            nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=1),
            nn.SELU(),
            nn.Conv2d(self.num_features, self.num_features, kernel_size=1)
        )

        # Mamba modules - high-resolution path (32x32 for 128-patch)
        self.GlobalMambaBlock1 = MambaLayerglobal(dim=self.num_features)
        self.LocalMambaBlock1 = MambaLayerlocal(dim=self.num_features)
        self.GlobalMambaBlock2 = MambaLayerglobal(dim=self.num_features)
        self.LocalMambaBlock2 = MambaLayerlocal(dim=self.num_features)

        # Downsampling
        self.conv1 = nn.Conv3d(
            self.num_features, self.num_features*2,
            kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)
        )

        # Mamba modules - low-resolution path (16x16 for 128-patch)
        self.GlobalMambaBlockLowRes1 = MambaLayerglobal(dim=self.num_features * 2)
        self.LocalMambaBlockLowRes1 = MambaLayerlocal(dim=self.num_features * 2)
        self.GlobalMambaBlockLowRes2 = MambaLayerglobal(dim=self.num_features * 2)
        self.LocalMambaBlockLowRes2 = MambaLayerlocal(dim=self.num_features * 2)
        self.GlobalMambaBlockLowRes3 = MambaLayerglobal(dim=self.num_features * 2)
        self.LocalMambaBlockLowRes3 = MambaLayerlocal(dim=self.num_features * 2)

        # Upsampling back to 32x32 (for 128-patch)
        self.upconv2 = nn.ConvTranspose3d(
            self.num_features*2, self.num_features,
            kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1),
            output_padding=(0, 1, 1)
        )

        # Final Mamba processing
        self.GlobalMambaBlock3 = MambaLayerglobal(dim=self.num_features)
        self.LocalMambaBlock3 = MambaLayerlocal(dim=self.num_features)
        self.GlobalMambaBlock4 = MambaLayerglobal(dim=self.num_features)
        self.LocalMambaBlock4 = MambaLayerlocal(dim=self.num_features)

        # AIM-VR reconstructor (fully reused)
        self.conv_before_upsample1 = nn.Conv3d(
            self.num_features, 128,
            kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )
        self.upsample1 = nn.PixelShuffle(2)  # 128 -> 32

        self.conv_before_upsample2 = nn.Conv3d(
            32, 64,
            kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )
        self.upsample2 = nn.PixelShuffle(2)  # 64 -> 16

        self.conv_last = nn.Conv3d(
            16, 3,
            kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )

    def forward(self, lqs, hilbert_curve_large_scale, hilbert_curve_small_scale):
        """Forward pass - fully reuses logic from flame_net.py, only adjusts optical flow method name"""
        b, t, c, h, w = lqs.size()

        # ====== Step 0: Compute all optical flows at once ======
        # Fix method name: call 128-patch-specific optical flow computation method
        all_flows = self._get_flows_all_scales(lqs)
        if all_flows is None and self.pa_frames <= 0:
            # Force disable all VRT TRSA switches
            use_trsa64a = use_trsa32 = use_trsa64b = False
            print(f"pa_frames={self.pa_frames}<=0, disabling all VRT TRSA modules")
        else:
            # Use configured switch states
            use_trsa64a = self.trsa64a
            use_trsa32 = self.trsa32
            use_trsa64b = self.trsa64b

        # ====== Step 1: AIM-VR feature extraction ======
        lqs_reshaped = lqs.reshape(-1, c, h, w)
        feats = self.feat_extract(lqs_reshaped)

        # ProjectionHead processing
        down1, down2, down3, down4 = self.head(feats)

        # Feature fusion
        down2_up = F.interpolate(down2, size=down1.size()[2:], mode='bilinear', align_corners=True)
        down3_up = F.interpolate(down3, size=down1.size()[2:], mode='bilinear', align_corners=True)
        down4_up = F.interpolate(down4, size=down1.size()[2:], mode='bilinear', align_corners=True)
        f = (down1 + down2_up + down3_up + down4_up) / 4
        f = self.refine(f) + f

        # Reshape to temporal features
        _, c, feat_h, feat_w = f.shape
        x_new = f.reshape(b, t, self.num_features, feat_h, feat_w)

        # ====== Step 2: Mamba M1 + FlowGuidedFusion64A ======
        M1 = self.GlobalMambaBlock1(x_new)
        M1 = self.LocalMambaBlock1(M1, hilbert_curve_large_scale)
        M1 = self.GlobalMambaBlock2(M1)
        M1 = self.LocalMambaBlock2(M1, hilbert_curve_large_scale)

        # FlowGuidedFusion64A enhancement (optional)
        if use_trsa64a:
            # Fix: safely extract optical flow to avoid KeyError
            flows_32 = None  # For 128-patch, the "64 position" is actually a 32x32 feature map
            if all_flows and 'flows_32' in all_flows:
                flows_32 = all_flows['flows_32']
            # If no valid optical flow, pass None tuple
            if flows_32:
                flows_tuple_64a = flows_32
            else:
                print("No valid flows_32 detected, flows_tuple_64a will pass (None, None)")
                flows_tuple_64a = (None, None)
            M1 = self._apply_flow_fusion(
                M1, flows_tuple_64a,
                flow_fusion=self.dmvrt_fusion_64a,
                advanced_fusion=self.advanced_fusion_64a
            )

        # ====== Step 3: Downsampling + Mamba M2 + FlowGuidedFusion32 ======
        x_down = rearrange(M1, 'n d c h w -> n c d h w')
        x_down = F.relu(self.conv1(x_down))
        x_down = rearrange(x_down, 'n c d h w -> n d c h w')

        M2 = self.GlobalMambaBlockLowRes1(x_down)
        M2 = self.LocalMambaBlockLowRes1(M2, hilbert_curve_small_scale)
        M2 = self.GlobalMambaBlockLowRes2(M2)
        M2 = self.LocalMambaBlockLowRes2(M2, hilbert_curve_small_scale)
        M2 = self.GlobalMambaBlockLowRes3(M2)
        M2 = self.LocalMambaBlockLowRes3(M2, hilbert_curve_small_scale)

        # FlowGuidedFusion32 enhancement (placed after M2 output, before Fre)
        if use_trsa32:
            # Fix: safely extract optical flow to avoid KeyError
            flows_16 = None  # For 128-patch, the "32 position" is actually a 16x16 feature map
            if all_flows and 'flows_16' in all_flows:
                flows_16 = all_flows['flows_16']
            # If no valid optical flow, pass None tuple
            if flows_16:
                flows_tuple_32 = flows_16
            else:
                print("No valid flows_16 detected, flows_tuple_32 will pass (None, None)")
                flows_tuple_32 = (None, None)

            M2 = self._apply_flow_fusion(
                M2, flows_tuple_32,
                flow_fusion=self.dmvrt_fusion_32,
                advanced_fusion=self.advanced_fusion_32
            )

        # Frequency enhancement processing (Fre fixed position unchanged)
        if self.fre_decoder:
            M2 = rearrange(M2, 'b t c h w -> (b t) c h w')
            M2 = self.fre2(lqs_reshaped, M2)
            M2 = rearrange(M2, '(b t) c h w -> b t c h w', b=b)

        # ====== Step 4: Upsampling + Mamba M3 + FlowGuidedFusion64B ======
        x_up = rearrange(M2, 'n d c h w -> n c d h w')
        x_up = F.relu(self.upconv2(x_up))
        if self.fre_decoder:
            M1 = rearrange(M1, 'n d c h w -> n c d h w')
            x_up = torch.cat([x_up, M1], 1)  # Concatenate along channel dimension [B, 512, T, H, W]
            x_up = self.conv_fre2_res(x_up)  # Channel alignment to 256 after residual concat
        x_up = rearrange(x_up, 'n c d h w -> n d c h w')

        M3 = self.GlobalMambaBlock3(x_up)
        M3 = self.LocalMambaBlock3(M3, hilbert_curve_large_scale)
        M3 = self.GlobalMambaBlock4(M3)
        M3 = self.LocalMambaBlock4(M3, hilbert_curve_large_scale)

        # FlowGuidedFusion64B enhancement (optional)
        if use_trsa64b:
            # Fix: safely extract optical flow to avoid KeyError
            flows_32 = None
            if all_flows and 'flows_32' in all_flows:
                flows_32 = all_flows['flows_32']
            # If no valid optical flow, pass None tuple
            if flows_32:
                flows_tuple_64b = flows_32
            else:
                print("No valid flows_32 detected, flows_tuple_64b will pass (None, None)")
                flows_tuple_64b = (None, None)

            M3 = self._apply_flow_fusion(
                M3, flows_tuple_64b,
                flow_fusion=self.dmvrt_fusion_64b,
                advanced_fusion=self.advanced_fusion_64b
            )

        # ====== Step 5: AIM-VR reconstruction ======
        x_re = rearrange(M3, 'n d c h w -> n c d h w')
        x_re = self._aimvr_reconstruction(x_re)
        final = self.conv_last(x_re)

        # Adjust dimension order to [B, T, C, H, W]
        final = final.permute(0, 2, 1, 3, 4)

        return final

    def _apply_flow_fusion(self, mamba_features, flows_tuple, flow_fusion, advanced_fusion):
        """Apply FlowGuidedFusion module - fully reuses logic from flame_net.py"""
        # Switch check: return original features directly when pa_frames<=0
        if self.pa_frames <= 0:
            return mamba_features

        # Check if flows_tuple is valid
        if flows_tuple is None or flows_tuple == (None, None):
            print(f"Invalid optical flow, skipping VRT processing")
            return mamba_features

        try:
            # Use precomputed optical flow
            flows_backward, flows_forward = flows_tuple

            # Further check if optical flow is None
            if flows_backward is None or flows_forward is None:
                print(f"Optical flow data is None, skipping VRT processing")
                return mamba_features

            # Step 1: FlowGuidedFusion smart processing
            vrt_features = flow_fusion(mamba_features, flows_backward, flows_forward)

            # Step 2: Compute flow_mask (optimized for weather degradation and occlusion)
            if self.use_flow_mask and flows_backward is not None and flows_forward is not None:
                flow_mask = flow_fusion._compute_flow_mask_weather_robust(flows_backward, flows_forward)
            else:
                flow_mask = None

            # Step 3: AdaptiveFusion - unified 2.5D convolution fusion
            fused_features = advanced_fusion(mamba_features, vrt_features, flow_mask)

        except Exception as e:
            print(f"VRT Stage processing failed: {e}, using original features")
            fused_features = mamba_features

        return fused_features

    def _get_flows_all_scales(self, x):
        """128-patch-specific optical flow computation - fix size mismatch issue

        Key fix: ensure optical flow sizes match feature map sizes
        - Input 128x128 -> ConvNeXt features 32x32 -> need 32x32 optical flow
        - After conv1 downsampling 16x16 -> need 16x16 optical flow
        - SpyNet output needs downsampling to corresponding sizes
        """
        b, t, c, h, w = x.size()

        # Return None if SpyNet unavailable or pa_frames<=0
        if not hasattr(self, 'spynet') or self.spynet is None or self.pa_frames <= 0:
            return None

        try:
            # 128-patch strategy: downsample to 64x64 for SpyNet computation
            spynet_input_size = 64  # SpyNet input size

            if torch.rand(1) < 0.01:  # 1% probability to print
                print(f"128-patch optical flow computation: input {h}x{w} -> SpyNet {spynet_input_size}x{spynet_input_size}")
                print(f"   Target flows: 32x32 feature map, 16x16 feature map")

            # 3D downsampling to SpyNet input size
            x_3d = x.permute(0, 2, 1, 3, 4)  # [B, 3, T, H, W]
            x_downsampled_3d = F.interpolate(
                x_3d,
                size=(t, spynet_input_size, spynet_input_size),
                mode='trilinear',
                align_corners=False
            )  # [B, 3, T, 64, 64]
            x_downsampled = x_downsampled_3d.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, 3, 64, 64]

            # SpyNet computation
            x_1 = x_downsampled[:, :-1, :, :, :].reshape(-1, c, spynet_input_size, spynet_input_size)
            x_2 = x_downsampled[:, 1:, :, :, :].reshape(-1, c, spynet_input_size, spynet_input_size)

            flows_backward_raw = self.spynet(x_1, x_2)
            flows_forward_raw = self.spynet(x_2, x_1)

            # Validate SpyNet return values
            if (flows_backward_raw is None or flows_forward_raw is None or
                not isinstance(flows_backward_raw, list) or not isinstance(flows_forward_raw, list) or
                len(flows_backward_raw) < 4 or len(flows_forward_raw) < 4):
                print(f"SpyNet returned invalid values")
                return None

            # Key fix: downsample SpyNet optical flow to target feature map sizes
            try:
                # SpyNet Level 0: 64x64 -> downsample to 32x32 (corresponds to 128-patch 32x32 feature map)
                flows_backward_64 = flows_backward_raw[0].reshape(b, t-1, 2, 64, 64)  # [B, T-1, 2, 64, 64]
                flows_forward_64 = flows_forward_raw[0].reshape(b, t-1, 2, 64, 64)

                # Downsample to 32x32
                flows_backward_64_3d = rearrange(flows_backward_64, 'b t c h w -> (b t) c h w')
                flows_forward_64_3d = rearrange(flows_forward_64, 'b t c h w -> (b t) c h w')

                fb_32 = F.interpolate(flows_backward_64_3d, size=(32, 32), mode='bilinear', align_corners=False)
                ff_32 = F.interpolate(flows_forward_64_3d, size=(32, 32), mode='bilinear', align_corners=False)

                fb_32 = rearrange(fb_32, '(b t) c h w -> b t c h w', b=b)  # [B, T-1, 2, 32, 32]
                ff_32 = rearrange(ff_32, '(b t) c h w -> b t c h w', b=b)  # [B, T-1, 2, 32, 32]

                # SpyNet Level 1: 32x32 -> downsample to 16x16 (corresponds to 128-patch 16x16 feature map)
                flows_backward_32 = flows_backward_raw[1].reshape(b, t-1, 2, 32, 32)  # [B, T-1, 2, 32, 32]
                flows_forward_32 = flows_forward_raw[1].reshape(b, t-1, 2, 32, 32)

                # Downsample to 16x16
                flows_backward_32_3d = rearrange(flows_backward_32, 'b t c h w -> (b t) c h w')
                flows_forward_32_3d = rearrange(flows_forward_32, 'b t c h w -> (b t) c h w')

                fb_16 = F.interpolate(flows_backward_32_3d, size=(16, 16), mode='bilinear', align_corners=False)
                ff_16 = F.interpolate(flows_forward_32_3d, size=(16, 16), mode='bilinear', align_corners=False)

                fb_16 = rearrange(fb_16, '(b t) c h w -> b t c h w', b=b)  # [B, T-1, 2, 16, 16]
                ff_16 = rearrange(ff_16, '(b t) c h w -> b t c h w', b=b)  # [B, T-1, 2, 16, 16]

                if torch.rand(1) < 0.01:  # 1% probability to print confirmation
                    print(f"Optical flow size fix: flows_32={fb_32.shape}, flows_16={fb_16.shape}")

                return {
                    'flows_32': (fb_32, ff_32),  # True 32x32 optical flow, corresponds to 32x32 feature map
                    'flows_16': (fb_16, ff_16)   # True 16x16 optical flow, corresponds to 16x16 feature map
                }

            except (IndexError, ValueError) as e:
                print(f"128-patch optical flow formatting failed: {e}")
                return None

        except Exception as e:
            print(f"128-patch SpyNet computation failed: {e}")
            return None

    def _aimvr_reconstruction(self, x):
        """AIM-VR reconstructor - fully reuses logic from flame_net.py"""
        # First upsampling: double feature map spatial size + channel reduction
        x = self.conv_before_upsample1(x)  # [B, 128, T, H, W]
        x = rearrange(x, 'n c d h w -> n d c h w')
        x = self.upsample1(x)  # [B, 32, T, H*2, W*2] (128/4=32)
        x = rearrange(x, 'n d c h w -> n c d h w')

        # Second upsampling: double feature map spatial size + channel processing
        x = self.conv_before_upsample2(x)  # [B, 64, T, H*2, W*2]
        x = rearrange(x, 'n c d h w -> n d c h w')
        x = self.upsample2(x)  # [B, 16, T, H*4, W*4] (64/4=16)
        x = rearrange(x, 'n d c h w -> n c d h w')

        return x

    def init_weights(self, pretrained=None, strict=False):
        """Initialize weights - fully reuses logic from flame_net.py"""
        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            logger.info(f'Load model from: {pretrained}')
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif hasattr(self.feat_extract, 'init_cfg') and self.feat_extract.init_cfg is not None:
            if isinstance(self.feat_extract.init_cfg.get('checkpoint'), (str, bytes)):
                self.feat_extract.init_weights()

    def _fix_ddp_checkpoint_issue(self):
        """Fix parameter duplicate marking issue in DDP training - fully reuses logic from flame_net.py"""
        print("Applying DDP fix: disabling all VRT checkpoint functionality...")

        checkpoint_disabled_count = 0

        for name, module in self.named_modules():
            # Disable VRT Stage checkpoint
            if hasattr(module, 'use_checkpoint_attn'):
                module.use_checkpoint_attn = False
                checkpoint_disabled_count += 1
            if hasattr(module, 'use_checkpoint_ffn'):
                module.use_checkpoint_ffn = False
                checkpoint_disabled_count += 1

            # Disable any torch.checkpoint related attributes
            if hasattr(module, 'checkpoint'):
                module.checkpoint = False
                checkpoint_disabled_count += 1

        print(f"DDP fix complete: disabled {checkpoint_disabled_count} checkpoint features")
