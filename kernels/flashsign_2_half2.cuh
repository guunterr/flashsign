#pragma once
#include <cublas_v2.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
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

namespace flashsign_2_half2 {

typedef __half fp16;
typedef __half2 fp162;

int ceil_div(int a, int b) {
    return (a / b) + (a % b != 0);
}


//64 registers for Q
//64 registers for O
//4 registers for S
//1 register for l
//4 registers for temporary work

template<const int NUM_THREADS, const int size>
__device__ void loadGMEMToSMEM(fp162 *src, fp162 *dst){
    for (uint i = 0; i < size; i+= 4 * NUM_THREADS)
    {
        float4 tmp = reinterpret_cast<float4 *>(&src[i + 4 * threadIdx.x])[0];
        reinterpret_cast<float4 *>(&dst[i + 4 * threadIdx.x])[0] = tmp;
    }
}

template<const int X, const int BX, const int BY, const int D>
__global__ void kernel(fp162 *Q, fp162 *K, fp162 *V, fp162 *O) {
    fp162 __shared__ KVs[2][BX * D];

    fp162 regQ[1 * D];
    fp162 regO[1 * D] = {};
    fp162 s2;
    fp16 l = 0;
    constexpr uint NUM_THREADS = BY;
    //Shuffle Q pointer to the right place
    Q += blockIdx.x * BY * D;
    O += blockIdx.x * BY * D;

    //thread gets its job
    const uint tix = threadIdx.x;

    //thread works with block to load Q
    //Load Q to SMEM in BX Chunks
    //Load those into regQ
    for (uint loadQBXBlock = 0; loadQBXBlock < BY; loadQBXBlock += 2 * BX)
    {   
        //Load a BXxD Chunk of Q into KVs
        loadGMEMToSMEM<NUM_THREADS, 2 * BX * D>(Q, &KVs[0][0]);
        __syncthreads();
        Q += 2 * D * BX;
        //Get that chunk into the appropriate register
        //We're eating some nasty SMEM conflicts here
        int rowIdx = (tix - loadQBXBlock);
        if(rowIdx >= 0 && rowIdx < 2 * BX){
            for (uint i = 0; i < D; i++) regQ[i] = KVs[rowIdx / BX][(rowIdx % BX) * D + i];
        }
        __syncthreads();
    }
    
    // Loop over X
    for (uint KVBlock = 0; KVBlock < X; KVBlock += BX)
    {
        //threads load part of K and V, size BX * D
        loadGMEMToSMEM<NUM_THREADS, BX * D>(K, &KVs[0][0]);
        loadGMEMToSMEM<NUM_THREADS, BX * D>(V, &KVs[1][0]);
        __syncthreads();
        //Shuffle K and V forwards
        K += BX * D;
        V += BX * D;
        ///Looping over BX
        #pragma unroll
        for (uint resIdx = 0; resIdx < BX; resIdx+=1)
        {
            //Initialise accumulator to zero
            s2 = __half2half2(CUDART_ZERO_FP16);

            //Calculate S = QK^T dot product for resIdx pair of elements
            for (uint dotIdx = 0; dotIdx < D; dotIdx++)
            {
                s2 += regQ[dotIdx] * KVs[0][resIdx * D + dotIdx];
            }
            //Calculate O = S V
            for (uint dotIdx = 0; dotIdx < D; dotIdx++)
            {
                regO[resIdx] += s2 * KVs[1][resIdx * D + dotIdx];
            }
            //Calculate l = sum(s^2)
            fp162 sqr = s2 * s2; //(s.x^2, s.y^2)
            l += (sqr.x + sqr.y);
        }
        __syncthreads();
    }
    fp162 norm_coeff = __half2half2(hrsqrt(l));
    //Thread normalises
    for (uint yIdx = 0; yIdx < D; yIdx+=8)
    {
        for (uint i = 0; i < 4; i++)
        {
            regO[yIdx + i] *= norm_coeff;
        }
        float4 tmp = reinterpret_cast<float4 *>(&regO[yIdx])[0];
        reinterpret_cast<float4 *>(&O[tix * D + yIdx])[0] = tmp;
    }
}

template<const int X, const int D>
void run_flashsign2_half(int Y, fp16 *Q, fp16 *K, fp16 *V, fp16 *O){
    constexpr int D_HALVED = D / 2;
    constexpr uint BY = 64;
    constexpr uint BX = 4;
    fp162* new_Q = (fp162 *)Q;
    fp162* new_K = (fp162 *)K;
    fp162* new_V = (fp162 *)V;
    fp162* new_O = (fp162 *)O;
    dim3 gridDim(ceil_div(Y, BY));
    dim3 blockDim(BY);
    kernel<X, BX, BY, D_HALVED><<<gridDim, blockDim>>>(new_Q, new_K, new_V, new_O);
}

template<const int X, const int D_HALVED>
void run_flashsign2_half2(int Y, fp162 *Q, fp162 *K, fp162 *V, fp162 *O){
    constexpr uint BY = 128;
    constexpr uint BX = 8;
    dim3 gridDim(ceil_div(Y, BY));
    dim3 blockDim(BY);
    kernel<X, BX, BY, D_HALVED><<<gridDim, blockDim>>>(Q, K, V, O);
}
}
