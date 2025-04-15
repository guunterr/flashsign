import os, torch
from importlib import reload
print(torch.version.cuda)

import kernels.flashattn16_128
reload(kernels.flashattn16_128)