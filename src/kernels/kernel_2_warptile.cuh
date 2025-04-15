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
__device__ void __inline_hint__ loadFromGMEM(bf16 *A, int lda_a, bf16 *As, bf16 *B, int lda_b, bf16 *Bs, int aInnerRow, int aInnerCol, int bInnerRow, int bInnerCol) {
    #pragma unroll
    for(uint loadOffset = 0; loadOffset < BM; loadOffset += strideA){
        bf16 tmp[8];
        float4 x = reinterpret_cast<float4 *>(&A[(aInnerRow + loadOffset) * lda_a + aInnerCol * 8])[0];
        memcpy(&tmp[0], &x, sizeof(bf16) * 8);
        //As transposed for better coalescing
        #pragma unroll
        for (uint i = 0; i < 8; i++)
        {
            As[(aInnerCol * 8 + i) * BM + aInnerRow + loadOffset] = tmp[i];
        }
    }
    #pragma unroll
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
        reinterpret_cast<float4 *>(&Bs[(bInnerRow + loadOffset) * BN + bInnerCol * 8])[0] 
            = reinterpret_cast<float4 *>(&B[(bInnerRow + loadOffset) * lda_b + bInnerCol * 8])[0];
    }
}

template<const int BM, const int BN, const int BK, const int TM, const int TN>
__device__ void __inline_hint__ matmulFromSMEM(bf16 *regM, bf16* regN, bf16* threadResults, const uint threadRow, const uint threadCol, const bf16 *As, const bf16 *Bs, bf16 *C) {
    #pragma unroll
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
        // Load block into registers
        #pragma unroll
        for (uint i = 0; i < TM; ++i) {
            regM[i] = As[dotIdx * BM + threadRow * TM + i];
        }
        #pragma unroll
        for (uint i = 0; i < TN; ++i) {
            regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
        }
        #pragma unroll
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                threadResults[resIdxM * TN + resIdxN] +=
                    regM[resIdxM] * regN[resIdxN];
            }
        }
    }
    return;
}

template<const int BM, const int BN, const int BK, const int TM, const int TN>
__device__ void __inline_hint__ storeToGMEM(bf16 *C, int ld_c, bf16 *threadResults, const uint threadRow, const uint threadCol) {
    #pragma unroll
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        #pragma unroll
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 8) {
            bf16 tmp[8];
            float4 x = reinterpret_cast<float4 *>(&C[(threadRow * TM + resIdxM) * ld_c + threadCol * TN + resIdxN])[0];
            memcpy(&tmp[0], &x, sizeof(bf16) * 8);
            #pragma unroll
            for (uint i = 0; i < 8; i++)
            {
                tmp[i] += threadResults[resIdxM * TN + resIdxN + i];
            }
            reinterpret_cast<float4 *>(&C[(threadRow * TM + resIdxM) * ld_c + threadCol * TN + resIdxN])[0] = reinterpret_cast<float4 *>(&tmp)[0];
        }
    }
    return;
}
__device__ void __inline_hint__ outerProductFromRegisters8x8(bf16 *regM, bf16* regN, bf16* threadResults) {
    #pragma unroll
    for (uint resIdxM = 0; resIdxM < 8; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < 8; ++resIdxN) {
            threadResults[resIdxM * 8 + resIdxN] +=
                regM[resIdxM] * regN[resIdxN];
        }
    }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN, const int WSUBM, const int WSUBN, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) kernel2(const int M, const int N, const int K, bf16 *A, bf16 *B, bf16 *C) {
    const uint cBlockRow = blockIdx.y;
    const uint cBlockCol = blockIdx.x;

    // We use BN/TN threads per column of the block of C we compute
    // Columns are contiguous in memory
    const uint threadCol = threadIdx.x % (BN / TN);
    const uint threadRow = threadIdx.x / (BN / TN);


    //Location of warp in threadblock
    const uint warpIdx = threadIdx.x / WARPSIZE;
    const uint warpRow = warpIdx / (BN / WN);
    const uint warpCol = warpIdx % (BN / WN);

    //Warptile Sizes
    constexpr uint warpSubRows = WM / WSUBM;
    constexpr uint warpSubCols = WN / WSUBN;

    __shared__ bf16 As[BM * BK];
    __shared__ bf16 Bs[BK * BN];

    // Move pointers to the start of the row of A, column of B and block of C
    A += cBlockRow * BM * K;
    B += cBlockCol * BN;
    C += cBlockRow * N * BM + cBlockCol * BN;

    // Indices that the thread loads into SMEM
    const uint aInnerCol = threadIdx.x % (BK / 8);
    const uint aInnerRow = threadIdx.x / (BK / 8);
    constexpr uint strideA = NUM_THREADS / (BK / 8);

    const uint bInnerCol = threadIdx.x % (BN / 8);
    const uint bInnerRow = threadIdx.x / (BN / 8);
    constexpr uint strideB = NUM_THREADS / (BN / 8);

    bf16 threadResults[warpSubRows][warpSubCols][TM * TN] = {0.0};
    bf16 regM[warpSubRows][TM] = {0.0};
    bf16 regN[warpSubCols][TN] = {0.0};
    for (uint block = 0; block < K; block += BK) {
        // Populate SMEM Caches
        loadFromGMEM<BM, BN, BK, strideA, strideB>(A, K, As, B, N, Bs, aInnerRow, aInnerCol, bInnerRow, bInnerCol);
        __syncthreads();

        // Advance blocktile
        A += BK;
        B += BK * N;
        //A_Tile starts at warpRow * WM * BK
        //B_Tile starts at warpCol * WN * BK
        for (uint WK = 0; WK < BK; WK++)
        {
            for (uint warpSubRow = 0; warpSubRow < warpSubRows; warpSubRow++)
            {
                float4 tmp = reinterpret_cast<float4 *>(&A[WK * BM + warpRow * WM + warpSubRow * WSUBM])[0];
                memcpy(&regM[warpSubRow][0], &tmp, sizeof(bf16) * 8);
            }    
            
            for (uint warpSubCol = 0; warpSubCol < WN; warpSubCol++)
            {
                float4 tmp = reinterpret_cast<float4 *>(&B[WK * BN + warpCol * WN + warpSubCol * WSUBN])[0];
                memcpy(&regN[warpSubCol][0], &tmp, sizeof(bf16) * 8);
            }

            for(uint warpSubRow = 0; warpSubRow < warpSubRows; warpSubRow++){
                for(uint warpSubCol = 0; warpSubCol < warpSubCols; warpSubCol++){
                    outerProductFromRegisters8x8(regM[warpSubRow], regN[warpSubCol], threadResults[warpSubRow][warpSubCol]);
                }
            }
        }
        
        __syncthreads();
    }
    // Write local registers to GMEM
    // storeToGMEM<BM, BN, BK, TM, TN>(C, N, threadResults, threadRow, threadCol);
}
}

using kernel2_warptile::kernel2;
