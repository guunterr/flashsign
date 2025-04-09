#pragma once
#include <cublas_v2.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_fp16.h>
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

namespace flashsign_1_basic {

typedef __half fp16;
typedef __half2 fp162;

int ceil_div(int a, int b) {
    return (a / b) + (a % b != 0);
}


template<const int X, const int BX, const int BY, const int D>
__global__ void kernel(fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    fp16 __shared__ Qs[BY * D];
    fp16 __shared__ KVs[D * BX];

    fp16 regQ[1 * D] = {0.0};
    fp16 regO[1 * D] = {0.0};
    fp16 regS[1 * BX] = {0.0};
    fp16 l = 0;
    constexpr uint NUM_THREADS = BY;
    //Shuffle Q pointer to the right place
    Q += blockIdx.x * BY * D;
    O += blockIdx.x * BY * D;

    //thread gets its job
    const uint tix = threadIdx.x;

    //thread works with block to load Q
    #pragma unroll
    for (uint loadQ = 0; loadQ < BY * D; loadQ += 8 * NUM_THREADS)
    {
        float4 tmp = reinterpret_cast<float4 *>(&Q[loadQ + 8 * tix])[0];
        reinterpret_cast<float4 *>(&Qs[loadQ + 8 * tix])[0] = tmp;
    }
    __syncthreads();
    for (uint i = 0; i < D; i++)
    {
        regQ[i] = Qs[tix * D + i];
    }
    //loop over X
    for (uint KVBlock = 0; KVBlock < X; KVBlock += BX)
    {
        //threads load part of K = BX * D
        for (uint loadK = 0; loadK < BX * D / (NUM_THREADS * 8); loadK += NUM_THREADS * 8)
        {
            fp16 tmp_K[8];
            float4 x = reinterpret_cast<float4 *>(&K[loadK + 8 * tix])[0];
            memcpy(&tmp_K[0], &x, sizeof(fp16)*8);

            uint rowIdx = (loadK + 8 * tix) / D;
            uint colIdx = (loadK + 8 * tix) % D;
            for (uint vectorIdx = 0; vectorIdx < 8; vectorIdx++)
            {
                KVs[BX * colIdx + rowIdx + vectorIdx] = tmp_K[vectorIdx];
            }
            
        }
        __syncthreads();
        //Shuffle K forwards
        K += BX * D;
        //Compute S = QK^T (1xBX)
        #pragma unroll
        for (uint dotIdx = 0; dotIdx < D; dotIdx++)
        {
            #pragma unroll
            for (uint K_CHUNK = 0; K_CHUNK < BX/8; K_CHUNK += 8)
            {
                fp16 tmp_k[8];
                float4 x = reinterpret_cast<float4 *>(&KVs[K_CHUNK + dotIdx * BX])[0];
                memcpy(&tmp_k[0], &x, sizeof(fp16)*8);
                for (uint i = 0; i < 8; i++)
                {
                    regS[i] += regQ[dotIdx] * tmp_k[i];
                }
            }
        }
        //thread accumulates normalisation value in registers
        for (uint i = 0; i < BX; i++)
        {
            l += regS[i] * regS[i];
        }
        __syncthreads();
        //threads load V (over K)
        for (uint loadV = 0; loadV < BX * D / (NUM_THREADS * 8); loadV += NUM_THREADS * 8)
        {
            fp16 tmp_V[8];
            float4 x = reinterpret_cast<float4 *>(&V[loadV + 8 * tix])[0];
            memcpy(&tmp_V[0], &x, sizeof(fp16)*8);

            uint rowIdx = (loadV + 8 * tix) / D;
            uint colIdx = (loadV + 8 * tix) % D;
            for (uint vectorIdx = 0; vectorIdx < 8; vectorIdx++)
            {
                KVs[BX * colIdx + rowIdx + vectorIdx] = tmp_V[vectorIdx];
            }
            
        }
        __syncthreads();
        //Shuffle V forwards BX Rows
        V += BX * D;
        //thread does DOT(S,V) = Y -> 1xD
        #pragma unroll
        for (uint dotIdx = 0; dotIdx < D; dotIdx++)
        {   
            for (uint V_CHUNK = 0; V_CHUNK < BX/8; V_CHUNK++)
            {
                fp16 tmp_V[8];
                float4 x = reinterpret_cast<float4 *>(&KVs[dotIdx * BX + V_CHUNK * 8])[0];
                memcpy(&tmp_V[0], &x, sizeof(fp16)*8);
                for (uint resIdx = 0; resIdx < 8; resIdx++)
                {
                    regO[dotIdx] += regS[V_CHUNK * 8 + resIdx] * tmp_V[resIdx];
                }
            }
            
        }
        __syncthreads();
    }
    l = hrsqrt(l);
    //Thread normalises
    for (uint yIdx = 0; yIdx < D; yIdx+=8)
    {
        for (uint i = 0; i < 8; i++)
        {
            regO[yIdx + i] *= l;
        }
        float4 tmp = reinterpret_cast<float4 *>(&regO[yIdx])[0];
        reinterpret_cast<float4 *>(&O[yIdx])[0] = tmp;
    }
}
template<const int X, const int D>
void run_flashsign1(int Y, fp16 *Q, fp16 *K, fp16 *V, fp16 *O){
    constexpr uint BY = 32;
    constexpr uint BX = 8;
    dim3 gridDim(ceil_div(Y, BY));
    dim3 blockDim(BY);
    kernel<X, BX, BY, D><<<gridDim, blockDim>>>(Q, K, V, O);
}
}
