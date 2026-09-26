# FLAME/AIM-VR Hilbert-sequence Mamba block (HSMB), built upon:
#   - state-spaces/mamba (https://github.com/state-spaces/mamba), Apache 2.0
#   - timm (https://github.com/huggingface/pytorch-image-models), Apache 2.0
# Hilbert 3D scanning: https://github.com/jakubcerveny/gilbert, BSD 2-Clause
# (see Hilbert3d.py header).
from mamba_ssm import Mamba
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from mmagic.models.editors.aimvsr.modules.Hilbert3d import Hilbert3d
import traceback
from typing import Optional, Tuple
import copy


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, nf, H, W):
        B, N, C = x.shape
        # print(f"DWConv input shape: B={B}, N={N}, C={C}")
        # print(f"Target reshape: nf={nf}, H={H}, W={W}")
        
        # Compute the actual required N value
        expected_N = nf * H * W
        if N != expected_N:
            print(f"Warning: N ({N}) != nf*H*W ({expected_N})")
            # Adjust N to match the expected dimension
            N = expected_N
            x = x[:, :N, :]
        
        # Rearrange dimensions
        x = x.transpose(1, 2).contiguous().reshape(B, C, nf, H, W)
        # print(f"After reshape: {x.shape}")
        
        # Apply 3D convolution
        x = self.dwconv(x)
        # print(f"After dwconv: {x.shape}")
        
        # Flatten and transpose back to the original format
        x = x.contiguous().flatten(2).transpose(1, 2)
        # print(f"Final output: {x.shape}")
        
        return x


class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        return self.conv(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, N, C] or [N, C]
        Returns:
            Tensor of shape [B, N, C] or [N, C]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MambaLayerglobal(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, mlp_ratio=4, drop=0., drop_path=0., act_layer=nn.GELU,
                 reverse=True):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v2",
            # use_fast_path=False,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.reverse = reverse
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        B, C, nf, H, W = x.shape

        assert C == self.dim

        n_tokens = x.shape[2:].numel()

        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)

        # Bi-Mamba layer
        x_mamba = x_flat + self.drop_path(self.mamba(self.norm1(x_flat)))
        x_mamba = x_mamba + self.drop_path(self.mlp(self.norm2(x_mamba)))

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        out = out.permute(0, 2, 1, 3, 4)

        return out


class MambaLayerlocal(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, mlp_ratio=4, drop=0., drop_path=0., act_layer=nn.GELU,
                 reverse=True):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v2",
            # use_fast_path=False,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.reverse = reverse
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, hilbert_curve):
        """Forward function of MambaLayer.
        Args:
            x (torch.Tensor): Input tensor with shape [B, nf, C, H, W]
            hilbert_curve (torch.Tensor): Hilbert curve indices with shape [hilbert_len]
        Returns:
            torch.Tensor: Output tensor with the same shape as input [B, nf, C, H, W]
        """
        x = x.permute(0, 2, 1, 3, 4)
        B, C, nf, H, W = x.shape

        if self.reverse:
            x = x.permute(0, 1, 3, 4, 2)
        assert C == self.dim

        img_dims = x.shape[2:]
        x_hw = x.flatten(2).contiguous()

        x_hil = x_hw.index_select(dim=-1, index=hilbert_curve)
        x_flat = x_hil.transpose(-1, -2)

        # Bi-Mamba layer
        x_mamba = x_flat + self.drop_path(self.mamba(self.norm1(x_flat)))
        x_mamba_out = x_mamba + self.drop_path(self.mlp(self.norm2(x_mamba)))

        out = x_mamba_out.transpose(-1, -2)

        sum_out = torch.zeros_like(out)
        hilbert_curve_re = repeat(hilbert_curve, 'hw -> b c hw', b=out.shape[0], c=out.shape[1])
        assert out.shape == hilbert_curve_re.shape

        sum_out.scatter_add_(dim=-1, index=hilbert_curve_re, src=out)
        sum_out = sum_out.reshape(B, C, *img_dims).contiguous()

        if self.reverse:
            sum_out = sum_out.permute(0, 1, 4, 2, 3)

        out = sum_out.permute(0, 2, 1, 3, 4)
        return out

#############################################################################################
########################Dynamic Parameters#############################################################
class FullyCon(nn.Module):
    """Feature extraction network, consistent with HAIR"""
    def __init__(self, num_box, in_features, mlp_depth=3):
        super().__init__()
        
        # Add dimension check and warning
        print(f"\nFullyCon initialization:")
        print(f"Input in_features: {in_features}")
        
        self.net = nn.Sequential()
        
        # First layer: map input features to a fixed dimension
        self.net.append(nn.Linear(in_features, in_features * 2))
        self.net.append(nn.GELU())
        ch = in_features * 2
        
        # Dynamically generate fully connected layers based on mlp_depth
        for i in range(mlp_depth-1):
            if i == mlp_depth-2:
                # Last layer maps to num_box
                self.net.append(nn.Linear(ch, num_box))
            else:
                # Intermediate layers gradually reduce dimension
                next_ch = ch // 2
                self.net.append(nn.Linear(ch, next_ch))
                self.net.append(nn.GELU())
                ch = next_ch

        # Print network structure
        print("\nFullyCon network structure:")
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                print(f"Layer {i}: in={layer.in_features}, out={layer.out_features}")
    
    def forward(self, x):
        """
        Args:
            x: Feature vector [B, in_features]
        Returns:
            Parameter weights [B, num_box]
        """
        return self.net(x)

class DynamicMambaLayerglobal(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, mlp_ratio=4, drop=0., drop_path=0., act_layer=nn.GELU,
                 reverse=True, num_box=4):
        super().__init__()
        self.feature_dim = dim
        
        # Use LayerNorm, applied on the channel dimension
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Mamba module configuration
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type="v2",
        )
        
        # Dynamic parameter generation network following HAIR
        self.param_net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.LayerNorm(dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 4, num_box)
        )
        
        # Get the parameter structure of the Mamba module
        param_info = []
        total_params = 0
        for name, param in self.mamba.named_parameters():
            if 'weight' in name or 'bias' in name:
                param_info.append({
                    'name': name,
                    'shape': param.shape,
                    'size': param.numel()
                })
                total_params += param.numel()
                print(f"Parameter {name}: shape {param.shape}, size {param.numel()}")
        
        # Initialize the parameter box
        self.kernel_box = nn.Parameter(torch.zeros(num_box, total_params))
        self.param_info = param_info
        
        # Compute the starting position of each parameter
        start_idx = 0
        self.param_indices = {}
        for info in param_info:
            size = info['size']
            self.param_indices[info['name']] = (start_idx, start_idx + size)
            start_idx += size
        
        # Initialize the parameter box with Mamba's initial parameters
        with torch.no_grad():
            for i in range(num_box):
                start_idx = 0
                for info in param_info:
                    name = info['name']
                    param = self.mamba.state_dict()[name]
                    size = info['size']
                    self.kernel_box[i, start_idx:start_idx + size] = param.reshape(-1)
                    start_idx += size
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        
    def forward(self, x, dynamic_feature=None):
        # x shape: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        
        # 1. Rearrange dimensions to apply LayerNorm
        x_reshaped = x.permute(0, 3, 4, 1, 2).reshape(-1, C)  # [B*H*W*T, C]
        x_norm = self.norm1(x_reshaped)  # [B*H*W*T, C]
        x_norm = x_norm.reshape(B, H, W, T, C).permute(0, 3, 4, 1, 2)  # [B, T, C, H, W]
        
        # 2. Generate dynamic parameters
        if dynamic_feature is not None:
            # Generate weight distribution from features
            # Use average pooling over spatial and temporal dimensions
            feature_pooled = F.adaptive_avg_pool3d(x_norm.permute(0, 2, 1, 3, 4), (1, 1, 1)).squeeze(-1).squeeze(-1).squeeze(-1)  # [B, C]
            coeffs = self.param_net(feature_pooled)  # [B, num_box]
            coeffs = F.softmax(coeffs, dim=-1)  # [B, num_box]
            
            # Dynamically combine weights
            mixed_params = torch.einsum('bn,np->bp', coeffs, self.kernel_box)  # [B, total_params]
            
            # Update Mamba parameters
            state_dict = self.mamba.state_dict()
            for info in self.param_info:
                name = info['name']
                start_idx, end_idx = self.param_indices[name]
                param_shape = info['shape']
                state_dict[name].data = mixed_params[:, start_idx:end_idx].reshape(-1, *param_shape[1:])
        
        # 3. Mamba processing
        x_flat = x_norm.permute(0, 1, 3, 4, 2).reshape(B, T*H*W, C)  # [B, L, C]
        x_mamba = self.mamba(x_flat)  # [B, L, C]
        
        # 4. Restore dimensions and apply residual connection
        x_mamba = x_mamba.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]
        x = x_norm + self.drop_path(x_mamba)
        
        return x

class DynamicMambaLayerlocal(MambaLayerlocal):
    """Local Mamba layer with dynamic parameters"""
    def __init__(self, dim, d_state=16, d_conv=4, expand=2,
                 mlp_ratio=4,
                 drop=0., drop_path=0., act_layer=nn.GELU, reverse=True,
                 num_box=8, param_dim=None):
        super().__init__(dim, d_state, d_conv, expand, mlp_ratio,
                        drop, drop_path, act_layer, reverse)
        
        self.dim = dim
        self.param_dim = param_dim if param_dim is not None else dim
        
        # Dynamic parameter generation network - use a smaller intermediate dimension
        mid_dim = min(self.param_dim, dim)  # Use the smaller dimension as the intermediate layer

        # Build the parameter generation network
        self.param_net = nn.Sequential(
            nn.Linear(self.param_dim, mid_dim),  # First adjust to the intermediate dimension
            nn.LayerNorm(mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, mid_dim),  # Keep the intermediate dimension unchanged
            nn.LayerNorm(mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, num_box)  # Finally output to num_box
        )
        
        print(f"\nDynamicMambaLayerlocal param_net structure:")
        print(f"Input dim: {self.param_dim}")
        print(f"Middle dim: {mid_dim}")
        print(f"Output dim: {num_box}")
        
        # Get the parameter structure of the Mamba module
        param_info = []
        total_params = 0
        for name, param in self.mamba.named_parameters():
            if 'weight' in name or 'bias' in name:
                param_info.append({
                    'name': name,
                    'shape': param.shape,
                    'size': param.numel()
                })
                total_params += param.numel()
                print(f"Parameter {name}: shape {param.shape}, size {param.numel()}")
        
        # Initialize the parameter box
        self.kernel_box = nn.Parameter(torch.zeros(num_box, total_params))
        self.param_info = param_info
        
        # Compute the starting position of each parameter
        start_idx = 0
        self.param_indices = {}
        for info in param_info:
            size = info['size']
            self.param_indices[info['name']] = (start_idx, start_idx + size)
            start_idx += size
        
        # Initialize the parameter box with Mamba's initial parameters
        with torch.no_grad():
            for i in range(num_box):
                start_idx = 0
                for info in param_info:
                    name = info['name']
                    param = self.mamba.state_dict()[name]
                    size = info['size']
                    self.kernel_box[i, start_idx:start_idx + size] = param.reshape(-1)
                    start_idx += size
        
    def forward(self, x, hilbert_curve, dynamic_feature=None):
        """
        Args:
            x: Input tensor [B, T, C, H, W]
            hilbert_curve: Hilbert curve indices [H*W*T]
            dynamic_feature: Dynamic features [B, C]
        """
        B, T, C, H, W = x.shape
        L = H * W * T
        
        # 1. Flatten and apply LayerNorm
        x_flat = x.view(B, T, C, H*W)        # [B, T, C, H*W]
        x_flat = x_flat.permute(0, 3, 1, 2)  # [B, H*W, T, C]
        x_flat = x_flat.reshape(B, -1, C)     # [B, T*H*W, C]
        x_norm = self.norm1(x_flat)           # [B, T*H*W, C]
        
        # 2. Apply Hilbert curve reordering
        hilbert_idx = hilbert_curve.view(1, -1, 1)        # [1, L, 1]
        hilbert_idx = hilbert_idx.expand(B, -1, C)        # [B, L, C]
        x_hil = torch.gather(x_norm, 1, hilbert_idx)      # [B, L, C]
        
        # 3. Generate dynamic parameters
        if dynamic_feature is not None:
            coeffs = self.param_net(dynamic_feature)  # [B, num_box]
            coeffs = F.softmax(coeffs, dim=-1)       # [B, num_box]
            
            # Dynamically combine weights
            mixed_params = torch.einsum('bn,np->bp', coeffs, self.kernel_box)  # [B, total_params]
            
            # Create a temporary Mamba module for forward pass
            temp_mamba = copy.deepcopy(self.mamba)
            
            # Update temporary Mamba parameters
            with torch.no_grad():
                start_idx = 0
                for info in self.param_info:
                    name = info['name']
                    start_idx, end_idx = self.param_indices[name]
                    param_shape = info['shape']
                    orig_param = temp_mamba.state_dict()[name]
                    
                    # Handle the batch dimension
                    if len(param_shape) >= 2:  # For weight matrices
                        param_data = mixed_params[:, start_idx:end_idx].reshape(B, *param_shape)
                        # Use the parameters from the first batch
                        param_data = param_data[0]
                    else:  # For bias vectors
                        param_data = mixed_params[:, start_idx:end_idx].reshape(B, -1)
                        param_data = param_data[0]
                    
                    # Ensure dimensions match
                    if param_data.shape == orig_param.shape:
                        temp_mamba.state_dict()[name].copy_(param_data)
                    else:
                        print(f"Warning: Parameter shape mismatch for {name}. Expected {orig_param.shape}, got {param_data.shape}")
            
            # Use the temporary Mamba for forward pass
            x_mamba = temp_mamba(x_hil)  # [B, L, C]
        else:
            x_mamba = self.mamba(x_hil)  # [B, L, C]
        
        # 5. Reverse reordering
        x_back = torch.zeros_like(x_norm)
        x_back.scatter_(1, hilbert_idx, x_mamba)
        
        # 6. MLP and residual connection
        x_mlp = self.mlp(x_back)  # Pass x_back directly, since Mlp has been updated to require no extra parameters
        x_out = x_back + self.drop_path(x_mlp)
        
        # 7. Reshape back to the original dimensions
        x_out = x_out.view(B, H*W, T, C)      # [B, H*W, T, C]
        x_out = x_out.permute(0, 2, 3, 1)     # [B, T, C, H*W]
        x_out = x_out.reshape(B, T, C, H, W)  # [B, T, C, H, W]
        
        return x_out

class HilbertCurveManager:
    """Hilbert curve manager for dynamically generating and caching Hilbert curves."""

    def __init__(self):
        self.curve_cache = {}  # Store Hilbert curves of different sizes
        
    @staticmethod
    def _create_hilbert_curve(H, W, nf):
        """Create a 3D Hilbert curve.

        Args:
            H (int): Height
            W (int): Width
            nf (int): Number of frames

        Returns:
            torch.Tensor: Hilbert curve indices
        """
        hilbert_curve = list(Hilbert3d(width=H, height=W, depth=nf))
        hilbert_curve = torch.tensor(hilbert_curve).long()
        return (
            hilbert_curve[:, 0] * W * nf +
            hilbert_curve[:, 1] * nf +
            hilbert_curve[:, 2]
        )
    
    def get_curve(self, H, W, nf, device=None):
        """Get the Hilbert curve of the specified size, creating it if it does not exist.

        Args:
            H (int): Height
            W (int): Width
            nf (int): Number of frames
            device (torch.device): Device

        Returns:
            torch.Tensor: Hilbert curve indices
        """
        key = (H, W, nf)
        if key not in self.curve_cache:
            curve = self._create_hilbert_curve(H, W, nf)
            self.curve_cache[key] = curve
        
        curve = self.curve_cache[key]
        if device is not None:
            curve = curve.to(device)
        return curve
    
    def clear_cache(self):
        """Clear the cache"""
        self.curve_cache.clear()

# Create a global Hilbert curve manager instance
hilbert_manager = HilbertCurveManager()
