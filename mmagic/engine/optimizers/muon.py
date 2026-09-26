# Copyright (c) OpenMMLab. All rights reserved.
"""
Muon optimizer wrapper and registry for MMEngine.

- If third-party package `muon` is available, delegate to `muon.Muon`.
- Otherwise, fall back to AdamW with a compatible interface, so training/tests won't break.

This module registers both 'Muon' and 'MuonOptimizer' into MMEngine's OPTIMIZERS
when imported.
"""

from __future__ import annotations

import warnings
from mmengine.registry import OPTIMIZERS
import torch
import torch.nn as nn

def register_muon_optimizer():
    """Register the MMEngine-compatible MuonWithAuxAdam hybrid optimizer."""
    try:
        # import the official hybrid optimizer
        from muon import MuonWithAuxAdam
        
        print("Found the official MuonWithAuxAdam hybrid optimizer!")
        
        # create a fully MMEngine-compatible wrapper
        class MuonWrapper:
            """Fully MMEngine-compatible wrapper of the Muon hybrid optimizer."""
            
            def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, 
                         betas=(0.9, 0.999), **kwargs):
                
                print(f"Initializing the MMEngine-compatible MuonWithAuxAdam:")
                print(f"   - Muon learning rate: {lr}")
                print(f"   - AdamW learning rate: {lr/20} (automatically set to 1/20)")
                
                # set the defaults attribute (required by MMEngine)
                self.defaults = {
                    'lr': lr,
                    'weight_decay': weight_decay,
                    'momentum': momentum,
                    'betas': betas
                }
                
                # collect all parameters
                all_params = []
                if hasattr(params, '__iter__'):
                    for group in params:
                        if isinstance(group, dict) and 'params' in group:
                            all_params.extend(group['params'])
                        elif isinstance(group, torch.nn.Parameter):
                            all_params.append(group)
                        else:
                            # assume it is an iterator of parameters
                            for param in group:
                                all_params.append(param)
                else:
                    all_params = [params]
                
                # automatically classify parameters: 2D -> Muon, others -> AdamW
                muon_params = []
                adamw_params = []
                
                for param in all_params:
                    if param.requires_grad:  # only process parameters that require gradients
                        if param.ndim >= 2:
                            muon_params.append(param)
                        else:
                            adamw_params.append(param)
                
                print(f"   Parameter classification results:")
                print(f"     - Muon parameters: {len(muon_params)} (2D weight matrices)")
                print(f"     - AdamW parameters: {len(adamw_params)} (1D parameters)")
                
                # build the param_groups format required by MuonWithAuxAdam
                param_groups = []
                
                # AdamW group (1D parameters)
                if adamw_params:
                    adamw_group = {
                        'params': adamw_params,
                        'lr': lr / 20,  # use a smaller learning rate for AdamW
                        'betas': betas,
                        'eps': 1e-8,
                        'weight_decay': weight_decay,
                        'use_muon': False
                    }
                    param_groups.append(adamw_group)
                
                # Muon group (2D parameters)
                if muon_params:
                    muon_group = {
                        'params': muon_params,
                        'lr': lr,
                        'momentum': momentum,
                        'weight_decay': weight_decay,
                        'use_muon': True
                    }
                    param_groups.append(muon_group)
                
                # create the actual MuonWithAuxAdam optimizer
                self.optimizer = MuonWithAuxAdam(param_groups)
                
                # set the attributes required by MMEngine
                self.param_groups = self.optimizer.param_groups
                self.state = self.optimizer.state
                
                print("MuonWithAuxAdam created successfully!")
                print(f"   - number of parameter groups: {len(self.param_groups)}")
                total_params = sum(len(group['params']) for group in self.param_groups)
                print(f"   - total number of parameters: {total_params}")
            
            def step(self, closure=None):
                """Perform an optimization step."""
                return self.optimizer.step(closure)
            
            def zero_grad(self, set_to_none: bool = False):
                """Clear gradients."""
                return self.optimizer.zero_grad(set_to_none)
            
            def state_dict(self):
                """Return the state dict."""
                return self.optimizer.state_dict()
            
            def load_state_dict(self, state_dict):
                """Load the state dict."""
                return self.optimizer.load_state_dict(state_dict)
            
            def add_param_group(self, param_group):
                """Add a parameter group."""
                return self.optimizer.add_param_group(param_group)
            
            def __repr__(self):
                return f"MuonWrapper(MuonWithAuxAdam with {len(self.param_groups)} groups)"
        
        # force registration, overriding any existing registration
        OPTIMIZERS.register_module(name='Muon', module=MuonWrapper, force=True)
        print("MMEngine-compatible MuonWithAuxAdam hybrid optimizer registered successfully!")
        return True
    except ImportError as e:
        # do not fall back to AdamW; raise the error directly
        raise ImportError(
            f"Failed to import the Muon optimizer: {e}\n"
            "Please install the Muon package: pip install git+https://github.com/KellerJordan/Muon.git"
        ) from e

# perform the registration
register_muon_optimizer()


__all__ = ['MuonOptimizer', 'register_muon_optimizer'] 