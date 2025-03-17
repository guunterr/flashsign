#pragma once
#include <cublas_v2.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#include <cassert>
#include <ctime>
#include <cuda/barrier>
#include <iostream>
#include <random>
#include <vector>

typedef __nv_bfloat16 bf16;

template <const int BM, const int BN, const int BK, const int TM>
__global__ void kernel4(int M, int N, int K, const bf16 *A, const bf16 *B, bf16 *C) {
    bf16 threadResults[TM] = {0.0};

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint threadCol = threadIdx.x % BN;
    const uint threadRow = threadIdx.x / BN;

    __shared__ bf16 As[BM * BK];
    __shared__ bf16 Bs[BK * BN];

    int Ablock = cRow * K * BM;
    int Bblock = cCol * BN;
    int Cblock = cRow * N * BM + cCol * BN;

    assert(BM * BK == blockDim.x);
    assert(BK * BN == blockDim.x);

    // Move along columns first then rows because cols are contiguous in memory
    const uint innerColA = threadIdx.x % BK;
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN;
    const uint innerRowB = threadIdx.x / BN;

    for (uint block = 0; block < K; block += BK) {
        // Fetch and store in GMEM
        As[innerColA + innerRowA * BK] = A[Ablock + innerColA + innerRowA * K];
        Bs[innerColB + innerRowB * BN] = B[Bblock + innerColB + innerRowB * N];
        __syncthreads();
        // Advance to next block
        Ablock += BK;
        Bblock += BK * N;

        // Calculate thread results (one column of C) block by block
        for (uint b_elem = 0; b_elem < BK; b_elem++) {
            bf16 tmp_B = Bs[b_elem * BN + threadCol];
            for (uint a_elem = 0; a_elem < TM; a_elem++) {
                threadResults[a_elem] += As[(threadRow * TM + a_elem) * BK + b_elem] * tmp_B;
            }
        }
        __syncthreads();
        for (uint i = 0; i < TM; i++)
        {
            C[Cblock + N *(threadRow * TM + i) + threadCol] += threadResults[i];
        }
        
    }
}