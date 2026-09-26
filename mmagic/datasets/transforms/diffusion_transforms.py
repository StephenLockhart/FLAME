# ==== Source: MMagic + WeatherDiffusion data processing ====
import numpy as np
import torch
from mmagic.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class ConcatInputs(BaseTransform):
    """
    Concatenate the conditional and target images into the input format of the diffusion model
    
    Args:
        keys (list): list of keys to concatenate, e.g. ['img', 'gt']
        output_key (str): output key name, defaults to 'inputs'
    """
    
    def __init__(self, keys, output_key='inputs'):
        self.keys = keys
        self.output_key = output_key
    
    def transform(self, results):
        """
        Args:
            results (dict): result dict containing img and gt
        
        Returns:
            dict: result dict with the concatenated inputs added
        """
        # get the data to concatenate
        data_list = []
        for key in self.keys:
            if key in results:
                data = results[key]
                # ensure the data is a numpy array
                if isinstance(data, torch.Tensor):
                    data = data.numpy()
                data_list.append(data)
        
        if len(data_list) > 0:
            # concatenate along the channel dimension (H, W, C) -> (H, W, C*N)
            concatenated = np.concatenate(data_list, axis=-1)
            results[self.output_key] = concatenated
        
        return results


@TRANSFORMS.register_module()
class NormalizeDiffusion(BaseTransform):
    """
    Normalization for diffusion models
    Normalize pixel values from [0, 255] to [-1, 1]
    """
    
    def __init__(self, keys=['inputs']):
        self.keys = keys
    
    def transform(self, results):
        """
        Args:
            results (dict): result dict containing the image data
        
        Returns:
            dict: normalized result dict
        """
        for key in self.keys:
            if key in results:
                data = results[key]
                # normalize to [-1, 1]
                if isinstance(data, np.ndarray):
                    data = data.astype(np.float32) / 127.5 - 1.0
                elif isinstance(data, torch.Tensor):
                    data = data.float() / 127.5 - 1.0
                results[key] = data
        
        return results


@TRANSFORMS.register_module()
class AddGaussianNoise(BaseTransform):
    """
    Add Gaussian noise for diffusion-model training
    
    Args:
        keys (list): list of keys to which noise is added
        noise_std (float): noise standard deviation
        prob (float): probability of adding noise
    """
    
    def __init__(self, keys=['img'], noise_std=0.1, prob=0.5):
        self.keys = keys
        self.noise_std = noise_std
        self.prob = prob
    
    def transform(self, results):
        """
        Args:
            results (dict): result dict containing the image data
        
        Returns:
            dict: result dict after adding noise
        """
        if np.random.random() < self.prob:
            for key in self.keys:
                if key in results:
                    data = results[key]
                    if isinstance(data, np.ndarray):
                        noise = np.random.normal(0, self.noise_std, data.shape).astype(data.dtype)
                        data = np.clip(data + noise, 0, 255)
                    elif isinstance(data, torch.Tensor):
                        noise = torch.randn_like(data) * self.noise_std
                        data = torch.clamp(data + noise, 0, 255)
                    results[key] = data
        
        return results


@TRANSFORMS.register_module()
class PrepareForDiffusion(BaseTransform):
    """
    Prepare the data format for diffusion models
    Convert the data to PyTorch tensors and permute the dimension order
    """
    
    def __init__(self, keys=['inputs']):
        self.keys = keys
    
    def transform(self, results):
        """
        Args:
            results (dict): result dict containing the image data
        
        Returns:
            dict: converted result dict
        """
        for key in self.keys:
            if key in results:
                data = results[key]
                
                # convert to tensors
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data)
                
                # permute the dimension order: (H, W, C) -> (C, H, W)
                if data.dim() == 3:
                    data = data.permute(2, 0, 1)
                
                # ensure the data type is float32
                data = data.float()
                
                results[key] = data
        
        return results 