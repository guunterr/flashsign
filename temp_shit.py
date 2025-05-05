import numpy as np
import torch

Q = torch.tensor([[(128 * i + j)/10 for j in range(16)] for i in range(16)], dtype=torch.float16)

for i in range(16):
    print(" ".join([f"{Q[i][j]:4.0f}" for j in range(16)]))
    
Q_0 = Q[:16][:8]
print(Q_0)