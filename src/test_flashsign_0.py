import os, torch
from importlib import reload
print(torch.version.cuda)

import kernels.flashsign_0_unfused
reload(kernels.flashsign_0_unfused)

def signed_attention(Q, K, V):
    S = torch.matmul(Q, torch.transpose(K, 0, 1))
    L = torch.linalg.norm(S, ord=2, dim=1, keepdim=True)
    S = S/L
    return torch.matmul(S, V)

torch.manual_seed(69)

Q = torch.randn((512,128), device='cuda', dtype=torch.half)
K = torch.randn((512,128), device='cuda', dtype=torch.half)
V = torch.randn((512,128), device='cuda', dtype=torch.half)

O_ref = signed_attention(Q, K, V)
O = kernels.flashsign_0_unfused.flashsign_unfused(Q, K, V)

print(torch.equal(O_ref, O))
print(O_ref.shape, O.shape)
print(O_ref.dtype, O.dtype)
print(O_ref[:,:])
print(O[:,:])