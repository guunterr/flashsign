import os, torch
from importlib import reload
print(torch.version.cuda)
import kernels.loadSave
X = torch.rand(1024, 1024, device='cuda', dtype=torch.half)
Y = kernels.loadSave.load_save(X)
print(torch.equal(X**2,Y))
print(X.shape, Y.shape)
print(X.dtype, Y.dtype)
print(X[0,0], Y[0,0])