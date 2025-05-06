import numpy as np
import torch

torch.set_printoptions(profile="full", sci_mode=False, linewidth=300, precision=3)

Q = torch.tensor([[(i + j)/100 for j in range(128)] for i in range(128)], dtype=torch.float16)
K = torch.tensor([[(i + j)/100 for j in range(128)] for i in range(32)], dtype=torch.float16)
    
# print(Q)
# print(torch.transpose(K,0,1))
print(torch.matmul(Q, torch.transpose(K,0,1)))