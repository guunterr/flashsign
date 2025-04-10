import torch
from torch.utils.cpp_extension import load_inline
from functools import wraps
from pathlib import Path
import inspect
import typing
from typing import Callable
import os

global cu_module, cu_dict, cpp_sources
cu_module = None
cu_dict = {}
cpp_sources = []
cpp_sources_toAdd = []
    
def cu_wrapper(cuda_name : str = ""):
    global cu_module, cu_dict, cpp_sources
    # The wrapper we will be applying
    def cu_remap[**P, T](func: Callable[P, T]) -> Callable[P, T]:
        # Returns a remapped function with the same name and docstring.
        @wraps(func)
        def cu_remapped(*args: P.args, **kwargs: P.kwargs) -> T:
            if func.__name__ in dir(cu_module):
                return cu_module.__getattribute__(func.__name__)(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        # Add it to the list of functions to be processed.
        cu_dict[cu_remapped.__name__] = ascii(cu_remapped.__doc__)
        if cuda_name:
            cpp_sources.append(cuda_name)
        else:
            cpp_sources_toAdd.append(cu_remapped.__name__)
        return cu_remapped
    return cu_remap

def cu_assign(extension_name):
    """Load inline requires:
     - cuda_sources
     - cpp_sources
     - functions
     """
    os.environ['CUDA_LAUNCH_BLOCKING']='1'
    os.environ['MAX_JOBS']='24'
    os.environ['TORCH_USE_CUDA_DSA']='1'
    os.environ['DS_BUILD_OPS']='1'
    global cu_module, cu_dict, cpp_sources
    path = Path(inspect.stack()[1].filename).parent / (extension_name + ".cu")
    with path.open() as file:
        cuda_sources = [file.read()]
    for k in cpp_sources_toAdd:
        loc = -1
        for cuda_source in cuda_sources:
            # find the function name
            loc = cuda_source.find(f" {k}(")
            if loc == -1:
                continue
            line_start = cuda_source.rfind("\n",0,loc)
            line_end = cuda_source.find("\n",loc)
            relevant_line = cuda_source[line_start:line_end]
            relevant_line = relevant_line.replace("{","").strip() +";"
            cpp_sources.append(relevant_line)
            break
        if loc == -1:
            raise Exception(f"{k} not found in any of the cuda sources!")
    cu_module = load_inline(
        cuda_sources=cuda_sources,
        with_cuda=None,
        cpp_sources=cpp_sources,
        functions=cu_dict,
        name=extension_name)