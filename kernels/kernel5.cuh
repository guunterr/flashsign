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

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void kernel5(const int M, const int N, const int K, const float *A, const float *B, float *C) {
    const uint cBlockRow = blockIdx.y;
    const uint cBlockCol = blockIdx.x;

    const uint totalResultsPerBlock = BM * BN;
    const uint totalResultsPerThread = TM * TN;

    // Res/Block / Res/Thread = Thread/Block which should be blockDim.x
    assert(totalResultsPerBlock / totalResultsPerThread == blockDim.x);

    // We use BN/TN threads per column of the block of C we compute
    // Columns are contiguous in memory
    const uint threadBlockCol = threadIdx.x % (BN / TN);
    const uint threadBlockRow = threadIdx.x / (BN / TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move pointers to the start of the row of A, column of B and block of C
    A += cBlockRow * BM * K;
    B += cBlockCol * BN;
    C += cBlockRow * N * BM + cBlockCol * BN;

    // Indices that the thread loads into SMEM
    const uint aInnerBlockCol = threadIdx.x % BK;
    const uint aInnerBlockRow = threadIdx.x / BK;
    const uint strideA = blockDim.x / BK;

    const uint bInnerBlockCol = threadIdx.x % BN;
    const uint bInnerBlockRow = threadIdx.x / BN;
    const uint strideB = blockDim.x / BN;

    float threadResults[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (uint block = 0; block < K; block += BK) {
        // Populate SMEM Caches
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            As[(aInnerBlockRow + loadOffset) * BK + aInnerBlockCol] =
                A[(aInnerBlockRow + loadOffset) * K + aInnerBlockCol];
        }
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            Bs[(bInnerBlockRow + loadOffset) * BN + bInnerBlockCol] =
                B[(bInnerBlockRow + loadOffset) * N + bInnerBlockCol];
        }
        __syncthreads();

        // Advance blocktile
        A += BK;
        B += BK * N;

        // Calculate thread's results to local registers
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Load block into registers
            for (uint i = 0; i < TM; i++) {
                regM[i] = As[(threadBlockRow * TM + i) * BK + dotIdx];
            }
            for (uint i = 0; i < TN; i++) {
                regN[i] = Bs[dotIdx * BN + threadBlockCol * TN + i];
                x
            }
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
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            C[(threadBlockRow * TM + resIdxM) * N + threadCol * TN + resIdxN] += threadResults[resIdxM * TN + resIdxN];
        }
    }
    

}
