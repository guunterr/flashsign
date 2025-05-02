import os, torch
import random
import sys
import numpy as np
from time import perf_counter
from importlib import reload
print(torch.version.cuda)

import kernels.flashsign_0_unfused
reload(kernels.flashsign_0_unfused)

def signed_attention(Q, K, V):
    S = torch.matmul(Q, torch.transpose(K, 0, 1))
    L = torch.linalg.norm(S, ord=2, dim=1, keepdim=True)
    S = S/L
    return torch.matmul(S, V)

def flashsign(Q, K, V):
    return kernels.flashsign_0_unfused.flashsign_unfused(Q, K, V)

def test_flashsign(kernel, reference, X, Y, epsilon=0.01):
    torch.manual_seed(69)

    print(f"Testing flashsign {sys.argv[1]}:")
    
    Q = torch.randn((Y,128), device='cuda', dtype=torch.half)
    K = torch.randn((X,128), device='cuda', dtype=torch.half)
    V = torch.randn((X,128), device='cuda', dtype=torch.half)

    O_ref = signed_attention(Q, K, V)
    O = flashsign(Q, K, V)

    correct_percentage = 100 * torch.count_nonzero(torch.abs(O_ref - O) < epsilon) / O.numel()

    print(f"{correct_percentage:.2f}% of elements are within {epsilon} of reference")
    
def benchmark_flashsign(kernel, reference, X, Y, warmups, runs):
    torch.manual_seed(69)
    
    print(f"Benchmarking flashsign {sys.argv[1]} over {runs} runs with {warmups} warmups")
    D = 128
    ref_times = []
    cuda_times = []
    for i in range(warmups + runs):
        Q = torch.randn((Y,D), device='cuda', dtype=torch.half)
        K = torch.randn((X,D), device='cuda', dtype=torch.half)
        V = torch.randn((X,D), device='cuda', dtype=torch.half)
        which_first = random.randint(0,1)
        if(which_first == 0):
            ref_timer_start = perf_counter()
            O_ref = reference(Q, K, V)
            torch.cuda.synchronize()
            ref_timer_stop = perf_counter()
            ref_times.append(ref_timer_stop - ref_timer_start)
            
            cuda_timer_start = perf_counter()
            O = kernel(Q,K,V)
            torch.cuda.synchronize()
            cuda_timer_stop = perf_counter()
            cuda_times.append(cuda_timer_stop - cuda_timer_start)
        else:
            cuda_timer_start = perf_counter()
            O = kernel(Q,K,V)
            torch.cuda.synchronize()
            cuda_timer_stop = perf_counter()
            cuda_times.append(cuda_timer_stop - cuda_timer_start)
            
            ref_timer_start = perf_counter()
            O_ref = reference(Q, K, V)
            torch.cuda.synchronize()
            ref_timer_stop = perf_counter()
            ref_times.append(ref_timer_stop - ref_timer_start)

    ref_times = ref_times[warmups:]
    cuda_times = cuda_times[warmups:]
    print(f"Pytorch time = {1000 * np.mean(ref_times):.1f}ms +- {1000 * np.std(ref_times):.1f}")
    print(f"Cuda time = {1000 * np.mean(cuda_times):.1f}ms +- {1000 * np.std(cuda_times):.1f}")

if __name__ == "__main__":
    if sys.argv[1] == "0":
        import kernels.flashsign_0_unfused
        reload(kernels.flashsign_0_unfused)
        def flashsign(Q, K, V):
            return kernels.flashsign_0_unfused.flashsign_unfused(Q, K, V)
    elif sys.argv[1] == "1":
        print("This shouldn't happen")
    elif sys.argv[1] == "2":
        import kernels.flashsign_2_half2
        reload(kernels.flashsign_2_half2)
        def flashsign(Q,K,V):
            return kernels.flashsign_2_half2.flashsign_2(Q, K, V)
    elif sys.argv[1] == "3":
        import kernels.flashsign_3_pytorch
        reload(kernels.flashsign_3_pytorch)
        def flashsign(Q,K,V):
            return kernels.flashsign_3_pytorch.flashsign_3(Q, K, V)
    elif sys.argv[1] == "4":
        import kernels.flashsign_kernel4
        reload(kernels.flashsign_kernel4)
        def flashsign(Q,K,V):
            return kernels.flashsign_kernel4.flashsign_4(Q, K, V)
    test_flashsign(flashsign, signed_attention, 1024, 1024)
    benchmark_flashsign(flashsign, signed_attention, int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))