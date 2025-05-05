import numpy as np
import torch

Q = torch.tensor([[(128 * i + j)/10 for j in range(16)] for i in range(16)], dtype=torch.float16)

print("Q shape", Q.shape)
for i in range(16):
    print(" ".join([f"{Q[i][j]:<5.1f}" for j in range(16)]))
    
Q_0 = Q[:, :8]
print("Q_0 = ", Q_0, Q_0.shape)
K_0 = Q[:8, 8:]
print("K_0 = ", K_0, K_0.shape)
print("KT = ", torch.transpose(K_0, 0,1))
S = torch.matmul(Q_0, torch.transpose(K_0, 0,1))
print("S = ", S, S.shape)