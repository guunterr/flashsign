import torch, typing, cuda_loader
from importlib import reload
reload(cuda_loader)

# The cuda_loader will search for a function with the name below in the .cu file with the same name as the .py
@cuda_loader.cu_wrapper()
def load_save(x: torch.Tensor) -> torch.Tensor:
    """The function does stuff.
    Args:
        input (torch.Tensor): The input tensor
    Returns:
        torch.Tensor: The ouput tensor
    """

# When the module is loaded, the function 'load_save' is rebinded to the compiled cuda function.
if "loadSave" in __name__:
    cuda_loader.cu_assign("loadSave")