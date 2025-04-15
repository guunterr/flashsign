import torch, typing, cuda_loader
from importlib import reload

reload(cuda_loader)

@cuda_loader.cu_wrapper()
def flashAttention():
    """The function does stuff.
    Args:
        input (torch.Tensor): The input tensor
    Returns:
        torch.Tensor: The ouput tensor
    """

if "flashattn16_128" in __name__:
    cuda_loader.cu_assign("flashattn16_128") 