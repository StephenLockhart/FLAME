import torch
print("CUDA available: ", torch.cuda.is_available())
print("Number of GPU devices: ", torch.cuda.device_count())
print("Current GPU device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
print(torch.__version__)  
print(torch.version.cuda)  

import torch
from mamba_ssm import Mamba
 
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("Using device: {}".format(device))

batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to("cuda")
y = model(x)
assert y.shape == x.shape
print('success')
print("Run successful, model output shape: {}".format(y.shape))