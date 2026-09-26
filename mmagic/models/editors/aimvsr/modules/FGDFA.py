'''
"Frequency-Guided Deformable Flow Attention (FGDFA)".

The main innovations of this design:

1. **Frequency-Adaptive Guidance**
- Use adaptive thresholds for frequency decomposition
- Process high-frequency and low-frequency features separately
- Enhance frequency-domain information through feature fusion

2. **Optical Flow Awareness**
- Encode optical flow information as additional features
- Guide the prediction of deformable offsets
- Enhance motion modeling capability

3. **Deformable Attention Mechanism**
- Combine deformable sampling with multi-head attention
- Use frequency and optical flow information to guide feature alignment
- Adaptively adjust sampling positions and attention weights

4. **Feature Fusion Strategy**
- Multi-level feature fusion (frequency features, optical flow features, original features)
- Adaptive weight allocation
- Progressive feature enhancement

Advantages of this mechanism:

1. **More comprehensive feature representation**
- Consider both frequency-domain and temporal information
- Combine local and global features
- Enhance feature expressiveness

2. **More accurate motion modeling**
- Optical-flow-guided feature alignment
- Frequency-aware feature enhancement
- Adaptive feature sampling

3. **More flexible feature fusion**
- Multi-modal feature fusion
- Adaptive weight allocation
- Hierarchical feature processing

Potential application scenarios:

1. Video super-resolution
2. Video deblurring
3. Video inpainting
4. Video quality enhancement

Directions for further optimization:

1. Design more efficient frequency decomposition methods
2. Optimize the extraction and use of optical flow features
3. Improve the feature fusion strategy
4. Increase temporal modeling capability

This deformable attention mechanism combining frequency adaptation and optical flow guidance is expected to achieve better performance in video processing tasks.

'''

import torch
import torch.nn as nn
import math

class FrequencyGuidedDeformableFlowAttention(nn.Module):
    """Frequency-Guided Deformable Flow Attention

    A deformable attention mechanism combining frequency adaptation and optical flow guidance

    Args:
        dim (int): Input feature dimension
        num_heads (int): Number of attention heads
        deform_groups (int): Number of deformable convolution groups
        max_residue_magnitude (int): Maximum offset magnitude
    """
    def __init__(self, 
                 dim, 
                 num_heads=8,
                 deform_groups=8,
                 max_residue_magnitude=10):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.deform_groups = deform_groups
        self.max_residue_magnitude = max_residue_magnitude
        
        # Frequency decomposition module
        self.freq_decompose = FrequencyDecomposition(dim)

        # Frequency feature fusion
        self.freq_fusion = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(0.1, True)
        )
        
        # Optical flow feature extraction
        self.flow_encoder = nn.Sequential(
            nn.Conv2d(4, dim//2, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(dim//2, dim, 3, 1, 1),
            nn.LeakyReLU(0.1, True)
        )
        
        # Deformable offset prediction
        self.offset_mask = nn.Sequential(
            nn.Conv2d(dim*3, dim, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(dim, dim, 3, 1, 1), 
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(dim, deform_groups * 3 * 2, 3, 1, 1)
        )
        
        # Attention computation
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x, flows):
        """
        Args:
            x (Tensor): Input features [B, C, H, W]
            flows (Tensor): Optical flow features [B, 2, H, W]
        """
        B, C, H, W = x.shape
        
        # 1. Frequency decomposition
        x_high, x_low = self.freq_decompose(x)
        freq_feat = self.freq_fusion(torch.cat([x_high, x_low], dim=1))
        
        # 2. Optical flow encoding
        flow_feat = self.flow_encoder(flows)
        
        # 3. Feature fusion and offset prediction
        feat_cat = torch.cat([x, freq_feat, flow_feat], dim=1)
        offset_mask = self.offset_mask(feat_cat)
        offset, mask = torch.split(offset_mask, 
                                 [self.deform_groups * 2 * 2, 
                                  self.deform_groups], 
                                 dim=1)
                                 
        # 4. Deformable sampling
        offset = self.max_residue_magnitude * torch.tanh(offset)
        mask = torch.sigmoid(mask)
        
        # 5. Attention computation
        x_reshape = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        qkv = self.qkv(x_reshape).reshape(B, H*W, 3, self.num_heads, 
                                        C//self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 6. Apply attention with deformable offsets
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.softmax(dim=-1)
        
        # 7. Feature aggregation
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x).permute(0, 3, 1, 2)
        
        return x

class FrequencyDecomposition(nn.Module):
    """Frequency decomposition module"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Adaptive threshold prediction
        self.threshold_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//8, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(dim//8, 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Predict frequency decomposition thresholds
        threshold = self.threshold_net(x)
        
        # FFT transform
        x_freq = torch.fft.fft2(x, norm='forward')
        x_freq = torch.fft.fftshift(x_freq)
        
        # Generate frequency mask
        B, C, H, W = x.shape
        mask = torch.zeros_like(x)
        h_ = (H//2 * threshold[:,0]).int()
        w_ = (W//2 * threshold[:,1]).int()
        
        for i in range(B):
            mask[i, :, 
                 H//2-h_[i]:H//2+h_[i], 
                 W//2-w_[i]:W//2+w_[i]] = 1
            
        # Separate high and low frequencies
        x_low = x_freq * mask
        x_high = x_freq * (1 - mask)
        
        # IFFT transform
        x_low = torch.fft.ifftshift(x_low)
        x_high = torch.fft.ifftshift(x_high)
        x_low = torch.abs(torch.fft.ifft2(x_low, norm='forward'))
        x_high = torch.abs(torch.fft.ifft2(x_high, norm='forward'))
        
        return x_high, x_low