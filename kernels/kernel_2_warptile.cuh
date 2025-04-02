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

namespace kernel2_warptile {

typedef __nv_bfloat16 bf16;

const int WARPSIZE = 32;

template <const int BM, const int BN, const int BK, const int strideA, const int strideB>
__device__ void loadFromGMEM(const bf16 *A, int lda_a, bf16 *As, const bf16 *B, int lda_b, bf16 *Bs, int aInnerBlockRow, int aInnerBlockCol, int bInnerBlockRow, int bInnerBlockCol) {
    #pragma unroll
    for(uint loadOffset = 0; loadOffset < BM; loadOffset += strideA){
        bf16 tmp[8];
        float4 x = reinterpret_cast<float4 *>(&A[(aInnerBlockRow + loadOffset) * lda_a + aInnerBlockCol * 8])[0];
        memcpy(&tmp[0], &x, sizeof(bf16) * 8);
        //As transposed for better coalescing
        #pragma unroll
        for (uint i = 0; i < 8; i++)
        {
            As[(aInnerBlockCol * 8 + i) * BM + aInnerBlockRow + loadOffset] = tmp[i];
        }
    }
    #pragma unroll
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
        reinterpret_cast<float4 *>(&Bs[(bInnerBlockRow + loadOffset) * BN + bInnerBlockCol * 8])[0] 
            = reinterpret_cast<float4 *>(&B[(bInnerBlockRow + loadOffset) * lda_b + bInnerBlockCol * 8])[0];
    }
}

template<const int BM, const int BN, const int BK, const int TM, const int TN>
__device__ void matmulFromSMEM(bf16 *regM, bf16* regN, bf16* threadResults, const bf16 *As, const bf16 *Bs, bf16 *C) {
    return;
}

template <const int BM, const int BN, const int BK, const int TM, const int TN, const int NUM_THREADS>
__global__ void kernel2(const int M, const int N, const int K, bf16 *A, bf16 *B,
                        bf16 *C) {
    const uint cBlockRow = blockIdx.y;
    const uint cBlockCol = blockIdx.x;

    // We use BN/TN threads per column of the block of C we compute
    // Columns are contiguous in memory
    const uint threadBlockCol = threadIdx.x % (BN / TN);
    const uint threadBlockRow = threadIdx.x / (BN / TN);

    __shared__ bf16 As[BM * BK];
    __shared__ bf16 Bs[BK * BN];

    // Move pointers to the start of the row of A, column of B and block of C
    A += cBlockRow * BM * K;
    B += cBlockCol * BN;
    C += cBlockRow * N * BM + cBlockCol * BN;

    // Indices that the thread loads into SMEM
    const uint aInnerBlockCol = threadIdx.x % (BK / 8);
    const uint aInnerBlockRow = threadIdx.x / (BK / 8);
    constexpr uint strideA = NUM_THREADS / (BK / 8);

    const uint bInnerBlockCol = threadIdx.x % (BN / 8);
    const uint bInnerBlockRow = threadIdx.x / (BN / 8);
    constexpr uint strideB = NUM_THREADS / (BN / 8);

    bf16 threadResults[TM * TN] = {0.0};
    bf16 regM[TM] = {0.0};
    bf16 regN[TN] = {0.0};
    for (uint block = 0; block < K; block += BK) {
        // Populate SMEM Caches
        loadFromGMEM<BM, BN, BK, strideA, strideB>(A, K, As, B, N, Bs, aInnerBlockRow, aInnerBlockCol, bInnerBlockRow, bInnerBlockCol);
        __syncthreads();

        // Advance blocktile
        A += BK;
        B += BK * N;

        // Calculate thread's results to local registers
        #pragma unroll
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Load block into registers
            #pragma unroll
            for (uint i = 0; i < TM; ++i) {
                regM[i] = As[dotIdx * BM + threadBlockRow * TM + i];
            }
            #pragma unroll
            for (uint i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadBlockCol * TN + i];
            }
            #pragma unroll
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] +=
                        regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }
    // Write local registers to GMEM
    #pragma unroll
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        #pragma unroll
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 8) {
            bf16 tmp[8];
            float4 x = reinterpret_cast<float4 *>(&C[(threadBlockRow * TM + resIdxM) * N + threadBlockCol * TN + resIdxN])[0];
            memcpy(&tmp[0], &x, sizeof(bf16) * 8);
            #pragma unroll
            for (uint i = 0; i < 8; i++)
            {
                tmp[i] += threadResults[resIdxM * TN + resIdxN + i];
            }
            reinterpret_cast<float4 *>(&C[(threadBlockRow * TM + resIdxM) * N + threadBlockCol * TN + resIdxN])[0] = reinterpret_cast<float4 *>(&tmp)[0];
        }
    }
}
}