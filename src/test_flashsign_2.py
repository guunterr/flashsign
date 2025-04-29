import os, torch
import sys
from importlib import reload
import numpy as np
from time import perf_counter
import random
print(torch.version.cuda)

import kernels.flashsign_2_half2
reload(kernels.flashsign_2_half2)

def print_full(tensor):
    torch.set_printoptions(profile="full")
    for row in range(tensor.shape[0]):
        print(list(map(lambda x : f"{float(x):.2f}", list(tensor[row,:]))))
    torch.set_printoptions(profile="default")
    print("\n\n")
def write(filename, tensor):
    with open(filename, "w") as f:
        for row in range(tensor.shape[0]):
            f.write(" ".join(list(map(lambda x : f"{float(x):.2f}", list(tensor[row,:])))) + "\n")


def signed_attention(Q, K, V):
    S = torch.matmul(Q, torch.transpose(K, 0, 1))
    L = torch.linalg.norm(S, ord=2, dim=1, keepdim=True)
    S = S/L
    return torch.matmul(S, V)

if __name__ == "__main__":

    torch.manual_seed(69)

    print("Compiled!")
    
    X = int(sys.argv[1])
    Y = int(sys.argv[2])
    D = 128
    ref_times = []
    cuda_times = []
    tries = int(sys.argv[3])
    for i in range(tries):
        Q = torch.randn((Y,D), device='cuda', dtype=torch.half)
        K = torch.randn((X,D), device='cuda', dtype=torch.half)
        V = torch.randn((X,D), device='cuda', dtype=torch.half)
        which_first = random.randint(0,1)
        if(which_first == 0):
            ref_timer_start = perf_counter()
            O_ref = signed_attention(Q, K, V)
            ref_timer_stop = perf_counter()
            ref_times.append(ref_timer_stop - ref_timer_start)
            
            cuda_timer_start = perf_counter()
            O = kernels.flashsign_2_half2.flashsign_2(Q, K, V)
            cuda_timer_stop = perf_counter()
            cuda_times.append(cuda_timer_stop - cuda_timer_start)
            
        else:
            cuda_timer_start = perf_counter()
            O = kernels.flashsign_2_half2.flashsign_2(Q, K, V)
            cuda_timer_stop = perf_counter()
            cuda_times.append(cuda_timer_stop - cuda_timer_start)
            
            ref_timer_start = perf_counter()
            O_ref = signed_attention(Q, K, V)
            ref_timer_stop = perf_counter()
            ref_times.append(ref_timer_stop - ref_timer_start)

    ref_times = ref_times[5:]
    cuda_times = cuda_times[5:]
    print(list(map(lambda x: round(x, 4), ref_times)))
    print(list(map(lambda x: round(x, 4), cuda_times)))
    print(f"Pytorch time = {1000 * np.mean(ref_times):.1f}ms +- {1000 * np.std(ref_times):.1f}")
    print(f"Cuda time = {1000 * np.mean(cuda_times):.1f}ms +- {1000 * np.std(cuda_times):.1f}")