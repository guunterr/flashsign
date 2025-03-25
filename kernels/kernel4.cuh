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


template <const int BM, const int BN, const int BK, const int TM>
__global__ void kernel4(const int M, const int N, const int K, const float *A, const float *B, float *C) {
    float threadResults[TM] = {0.0};

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint threadCol = threadIdx.x % BN;
    const uint threadRow = threadIdx.x / BN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * K * BM;
    B += cCol * BN;
    C += cRow * N * BM + cCol * BN;

    assert(BM * BK == blockDim.x);
    assert(BK * BN == blockDim.x);

    // Move along columns first then rows because cols are contiguous in memory
    const uint innerColA = threadIdx.x % BK;
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN;
    const uint innerRowB = threadIdx.x / BN;

    #pragma unroll
    for (uint block = 0; block < K; block += BK) {
        // Fetch and store in SMEM
        As[innerColA + innerRowA * BK] = A[innerColA + innerRowA * K];
        Bs[innerColB + innerRowB * BN] = B[innerColB + innerRowB * N];
        __syncthreads();
        // Advance to next block
        A += BK;
        B += BK * N;

        // Calculate thread results (one column of C) block by block
        #pragma unroll
        for (uint b_elem = 0; b_elem < BK; ++b_elem) {
            float tmp_B = Bs[b_elem * BN + threadCol];
            #pragma unroll
            for (uint a_elem = 0; a_elem < TM; ++a_elem) {
                threadResults[a_elem] += As[(threadRow * TM + a_elem) * BK + b_elem] * tmp_B;
            }
        }
        __syncthreads();
        
        
    }
    #pragma unroll
    for (uint i = 0; i < TM; ++i)
    {
        C[N *(threadRow * TM + i) + threadCol] = threadResults[i] + C[N *(threadRow * TM + i) + threadCol];
    }
}