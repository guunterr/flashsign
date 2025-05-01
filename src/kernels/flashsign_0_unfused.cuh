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

//Unashamed to say I vibe coded this
//It tried transposing K and then retransposing it in the call to CUBLAS
//Win for humanity over the machines!
namespace flashsign_0_kernel {

    typedef __half fp16;
    typedef __half2 fp162;
    
    int ceil_div(int a, int b) {
        return (a / b) + (a % b != 0);
    }
    
    __global__ void compute_row_norm_rsqrt(const fp16* S, fp16* norms, int Y, int X) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        float sum = 0.0;
        for (int col = 0; col < X; col++) {
            sum += __half2float(__hmul(S[row * X + col], S[row * X + col]));
        } 
        norms[row] = __float2half(rsqrt(sum));
    }
    
    __global__ void normalise_rows(fp16* S, const fp16* norms, int Y, int X){
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        for (int col = 0; col < X; col++) {
            S[row * X + col] = __hmul(S[row * X + col], norms[row]);
        }
    }
    
    __global__ void compute_QKT(const fp16* Q, const fp16* K, fp16* S, int Y, int X, int D) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (row < Y && col < X) {
            fp16 sum = __float2half(0.0f);
            for (int k = 0; k < D; k++) {
                sum = __hadd(sum, __hmul(Q[row * D + k], K[col * D + k]));
            }
            S[row * X + col] = sum;
        }
    }
    
    __global__ void compute_SV(const fp16* S, const fp16* V, fp16* O, int Y, int X, int D) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y * blockDim.y + threadIdx.y;
        if (row < Y && col < D) {
            fp16 sum = __float2half(0.0f);
            for (int k = 0; k < X; k++) {
                sum = __hadd(sum, __hmul(S[row * X + k], V[k * D + col]));
            }
            O[row * D + col] = sum;
        }
    }
    
    // Main function to compute (row-wise L2 norm (Q K^T)) V without explicit transposes
    void run_unfused_flashsign(const fp16* Q, const fp16* K, const fp16* V, fp16* O, int Y, int X, int D) {
        printf("Running unfused_flashsign\n");
    
        // Step 1: Compute S = Q K^T
        fp16* S;
        cudaMalloc(&S, X * Y * sizeof(fp16));
        fp16* norms;
        cudaMalloc(&norms, Y * sizeof(fp16));
    
    
        //Computing S = Q K^T which is YxX
        //We have Q^T and K^T. So transpose Q.
        compute_QKT<<<dim3(ceil_div(Y, 32), ceil_div(X, 32)), dim3(32, 32)>>>(Q, K, S, Y, X, D);
        cudaDeviceSynchronize();
    
        // Step 2: Compute L2 norms of S's rows
        compute_row_norm_rsqrt<<<ceil_div(Y, 256), 256>>>(S, norms, Y, X);
        cudaDeviceSynchronize();
    
        // Step 3: Normalize rows of S
        normalise_rows<<<ceil_div(Y, 256), 256>>>(S, norms, Y, X);
        cudaDeviceSynchronize();
    
        // Step 4: Compute output = S_normalized * V which is YxD
        //We have S_norm_T and V_T when read column-major.
        //So we can compute S_norm_T_T * V_T_T = S_norm V
        compute_SV<<<dim3(ceil_div(Y, 32), ceil_div(D, 32)), dim3(32, 32)>>>(S, V, O, Y, X, D);
        cudaDeviceSynchronize();
    
        // Clean up
        cudaFree(S);
        cudaFree(norms);
    }
    
}
