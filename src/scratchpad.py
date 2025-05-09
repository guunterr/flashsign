import numpy as np
import torch

torch.set_printoptions(profile="full", sci_mode=False, linewidth=400, precision=3)

X = 128
Y = 128
D = 128

Q = torch.tensor([[(i + j)/10000 for j in range(D)] for i in range(Y)], dtype=torch.float16)
K = torch.tensor([[(i + j)/10000 for j in range(D)] for i in range(X)], dtype=torch.float16)
V = torch.tensor([[(i + j)/10000 for j in range(D)] for i in range(X)], dtype=torch.float16)
    
# print(Q)
# print(torch.transpose(K,0,1))
S = torch.matmul(Q, torch.transpose(K,0,1))
L = torch.linalg.norm(S, ord=2, dim=1, keepdim=True)
S_norm = S/L
O = torch.matmul(S_norm, V)
print(f"Q ({Q.shape[0]}x{Q.shape[1]}) = ", Q)
print(f"K {K.shape[0]}x{K.shape[1]}) = ", K)
print(f"V {V.shape[0]}x{V.shape[1]}) = ", V)
print(f"S {S.shape[0]}x{S.shape[1]}) = ", S)
print(f"L {L.shape[0]}x{L.shape[1]}) = ", L)
print(f"S_norm = {S_norm.shape[0]}x{S_norm.shape[1]} = ", S_norm)
print(f"O {O.shape[0]}x{O.shape[1]}) = ", O)