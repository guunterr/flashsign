import torch, typing, cuda_loader
from importlib import reload

reload(cuda_loader)

@cuda_loader.cu_wrapper()
def flashsign_2():
    """The function does stuff.
    Args:
        input (torch.Tensor): The input tensor
    Returns:
        torch.Tensor: The ouput tensor
    """

if "flashsign_2_half2" in __name__:
    cuda_loader.cu_assign("flashsign_2_half2") 