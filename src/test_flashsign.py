import os, torch
import random
import sys
import numpy as np
from time import perf_counter
from importlib import reload
print(torch.version.cuda)

def signed_attention(Q, K, V):
    S = torch.matmul(Q, torch.transpose(K, 0, 1))
    L = torch.linalg.norm(S, ord=2, dim=1, keepdim=True)
    S = S/L
    return torch.matmul(S, V)


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
    
def benchmark_flashsign(kernel, X, Y, warmups, runs):
    torch.manual_seed(69)
    
    
    D = 128
    print(f"Benchmarking flashsign {sys.argv[1]} over {runs} runs with {warmups} warmups, X = {X}, Y = {Y}, D = {D}")
    times = []
    for i in range(warmups + runs):
        Q = torch.randn((Y,D), device='cuda', dtype=torch.half)
        K = torch.randn((X,D), device='cuda', dtype=torch.half)
        V = torch.randn((X,D), device='cuda', dtype=torch.half)

        cuda_timer_start = perf_counter()
        O = kernel(Q,K,V)
        torch.cuda.synchronize()
        cuda_timer_stop = perf_counter()
        times.append(cuda_timer_stop - cuda_timer_start)


    times = times[warmups:]
    print(f"Kernel time = {1000 * np.mean(times):.1f}ms +- {1000 * np.std(times):.1f}")

if __name__ == "__main__":
    if sys.argv[1] == "0":
        def flashsign(Q, K, V):
            return signed_attention(Q,K,V)
    # elif sys.argv[1] == "1":
    #     import kernels.flashsign_0_unfused
    #     reload(kernels.flashsign_0_unfused)
    #     def flashsign(Q, K, V):
    #         return kernels.flashsign_0_unfused.flashsign_unfused(Q, K, V)

    # elif sys.argv[1] == "2":
    #     import kernels.flashsign_2_half2
    #     reload(kernels.flashsign_2_half2)
    #     def flashsign(Q,K,V):
    #         return kernels.flashsign_2_half2.flashsign_2(Q, K, V)
    # elif sys.argv[1] == "3":
    #     import kernels.flashsign_3_pytorch
    #     reload(kernels.flashsign_3_pytorch)
    #     def flashsign(Q,K,V):
    #         return kernels.flashsign_3_pytorch.flashsign_3(Q, K, V)
    # elif sys.argv[1] == "4":
    #     with open("./kernels/flashsign_kernel4_torchbind.cu", "r") as torch_binding_file:
    #         with open("./kernels/flashsign_kernel4_cuda.cu", "r") as kernel_file:
    #             with open("./kernels/flashsign_kernel4.cu", "w") as write_file:
    #                 write_file.write("\n" + kernel_file.read() + "\n\n" + torch_binding_file.read())
    #     #     print(kernel_file.read())
    #     import kernels.flashsign_kernel4
    #     reload(kernels.flashsign_kernel4)
    #     def flashsign(Q,K,V):
    #         return kernels.flashsign_kernel4.flashsign_4(Q, K, V)
    # test_flashsign(flashsign, signed_attention, 1024, 1024)
    benchmark_flashsign(flashsign, 108 * 128 * int(sys.argv[2]), 108 * 128 * int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))