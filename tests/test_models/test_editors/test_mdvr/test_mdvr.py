# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmagic.registry import MODELS
from mmagic.structures import DataSample


class TestMDVR:
    
    @classmethod
    def setup_class(cls):
        cls.default_config = dict(
            type='MDVR',
            generator=dict(
                type='MDVRNet',
                in_channels=6,
                out_channels=3,
                model_channels=64,
                num_res_blocks=2,
                attention_resolutions=[8, 16],
                channel_mult=[1, 2, 4, 8],
                num_frames=5,
                fre_decoder=True,
                use_hilbert_3d=True,
                block_mode_pattern=['global', 'local'],
            ),
            pixel_loss=dict(
                type='CharbonnierLoss',
                loss_weight=1.0,
                reduction='mean'
            ),
            perceptual_loss=dict(
                type='PerceptualLoss',
                layer_weights={
                    '3': 1.0,
                    '8': 1.0,
                    '15': 0.5,
                },
                vgg_type='vgg16',
                criterion='mse',
                perceptual_weight=0.02,
                style_weight=0,
                norm_img=False,
                pretrained='torchvision://vgg16'
            ),
            contrast_loss=dict(
                type='ContrastLoss',
                loss_weight=0.05,
                reduction='mean'
            ),
            diffusion_config=dict(
                type='EditDDPMScheduler',
                num_train_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule='linear',
                variance_type='fixed_small',
                clip_sample=True,
            ),
            inference_scheduler_config=dict(
                type='EditDDIMScheduler',
                num_train_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule='linear',
                clip_sample=True,
                set_alpha_to_one=True,
            ),
            scale_factor=1,
            num_input_frames=5,
            train_cfg=dict(
                video_clip_length=5,
                temporal_stride=1,
                spatial_size=(256, 256),
                gradient_checkpointing=True,
                mixed_precision=True,
                accumulation_steps=4,
            ),
            test_cfg=dict(
                scale_factor=1,
                num_inference_steps=25,
                tile=[5, 256, 256],
                tile_overlap=[4, 192, 192],
            ),
            data_preprocessor=dict(
                type='DataPreprocessor',
                mean=[0.0, 0.0, 0.0],
                std=[255.0, 255.0, 255.0],
            ),
        )
    
    def test_mdvr_init(self):
        """Test MDVR initialization."""
        model = MODELS.build(self.default_config)
        assert model is not None
        assert hasattr(model, 'generator')
        assert hasattr(model, 'pixel_loss')
        assert hasattr(model, 'perceptual_loss')
        assert hasattr(model, 'contrast_loss')
        assert hasattr(model, 'training_scheduler')
        assert hasattr(model, 'inference_scheduler')
    
    def test_mdvr_forward_train(self):
        """Test MDVR forward training."""
        model = MODELS.build(self.default_config)
        
        # Create dummy input data
        B, T, C, H, W = 1, 5, 3, 64, 64
        inputs = torch.randn(B, T, C, H, W)
        
        # Create dummy data samples
        gt_img = torch.randn(B, T, C, H, W)
        data_samples = DataSample()
        data_samples.gt_img = gt_img
        
        # Test forward training
        losses = model.forward_train(inputs, [data_samples])
        
        assert isinstance(losses, dict)
        assert 'loss_diffusion' in losses
        assert 'loss_pix' in losses
        assert 'loss_perceptual' in losses
        assert 'loss_contrast' in losses
        
        # Check loss values are tensors
        for loss_name, loss_value in losses.items():
            assert torch.is_tensor(loss_value)
            assert loss_value.requires_grad
    
    def test_mdvr_forward_inference(self):
        """Test MDVR forward inference."""
        model = MODELS.build(self.default_config)
        model.eval()
        
        # Create dummy input data
        B, T, C, H, W = 1, 5, 3, 64, 64
        inputs = torch.randn(B, T, C, H, W)
        
        with torch.no_grad():
            predictions = model.forward_inference(inputs)
        
        assert isinstance(predictions, DataSample)
        assert hasattr(predictions, 'pred_img')
        assert predictions.pred_img.shape == (B, T, C, H, W)
    
    def test_mdvr_forward_tensor(self):
        """Test MDVR forward tensor."""
        model = MODELS.build(self.default_config)
        model.eval()
        
        # Create dummy input data  
        B, T, C, H, W = 1, 5, 3, 64, 64
        inputs = torch.randn(B, T, C, H, W)
        
        with torch.no_grad():
            output = model.forward_tensor(inputs)
        
        assert torch.is_tensor(output)
        assert output.shape == (B, T, C, H, W)
    
    def test_hilbert_curves_initialization(self):
        """Test Hilbert curves are properly initialized."""
        model = MODELS.build(self.default_config)
        
        assert hasattr(model, 'hilbert_large')
        assert hasattr(model, 'hilbert_small')
        assert torch.is_tensor(model.hilbert_large)
        assert torch.is_tensor(model.hilbert_small)
        
        # Check that Hilbert curves have correct length
        T = model.num_input_frames
        assert len(model.hilbert_large) <= 64 * 64 * T
        assert len(model.hilbert_small) <= 32 * 32 * T
    
    def test_different_input_sizes(self):
        """Test MDVR with different input sizes."""
        model = MODELS.build(self.default_config)
        model.eval()
        
        # Test different spatial sizes
        sizes = [(32, 32), (64, 64), (128, 128)]
        
        for H, W in sizes:
            B, T, C = 1, 5, 3
            inputs = torch.randn(B, T, C, H, W)
            
            with torch.no_grad():
                output = model.forward_tensor(inputs)
            
            assert output.shape == (B, T, C, H, W)
    
    @pytest.mark.parametrize('num_frames', [3, 5, 7])
    def test_different_temporal_sizes(self, num_frames):
        """Test MDVR with different temporal sizes."""
        config = self.default_config.copy()
        config['num_input_frames'] = num_frames
        config['generator']['num_frames'] = num_frames
        
        model = MODELS.build(config)
        model.eval()
        
        B, T, C, H, W = 1, num_frames, 3, 64, 64
        inputs = torch.randn(B, T, C, H, W)
        
        with torch.no_grad():
            output = model.forward_tensor(inputs)
        
        assert output.shape == (B, T, C, H, W)
    
    def test_mdvr_net_separate(self):
        """Test MDVRNet separately."""
        from mmagic.models.editors.mdvr import MDVRNet
        
        net = MDVRNet(
            in_channels=6,
            out_channels=3,
            model_channels=64,
            num_res_blocks=2,
            attention_resolutions=[8, 16],
            channel_mult=[1, 2, 4, 8],
            num_frames=5,
            fre_decoder=True,
            use_hilbert_3d=True,
            block_mode_pattern=['global', 'local'],
        )
        
        # Test diffusion mode
        B, C, T, H, W = 1, 6, 5, 64, 64
        x = torch.randn(B, C, T, H, W)
        timesteps = torch.randint(0, 1000, (B,))
        hilbert_large = torch.randint(0, 64*64*5, (1000,))
        hilbert_small = torch.randint(0, 32*32*5, (500,))
        
        output = net(x, timesteps, hilbert_large, hilbert_small)
        assert output.shape == (B, T, 3, H, W)
        
        # Test direct mode (no timesteps)
        x_direct = torch.randn(B, 3, T, H, W)
        output_direct = net(x_direct, None, hilbert_large, hilbert_small)
        assert output_direct.shape == (B, T, 3, H, W) 