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
namespace flashsign_0_unfused {

typedef __half fp16;
typedef __half2 fp162;

int ceil_div(int a, int b) {
    return (a / b) + (a % b != 0);
}

__global__ void compute_row_norm_rsqrt(const fp16* S, fp16* norms, int Y, int X) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    fp16 sum = 0.0;
    for (int col = 0; col < X; col++) {
        sum += S[row * X + col] * S[row * X + col];
    } 
    norms[row] = hrsqrt(sum);
}

__global__ void normalise_rows(fp16* S, const fp16* norms, int Y, int X){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    for (int col = 0; col < X; col++) {
        S[row * X + col] *= norms[row];
    }
}

// Main function to compute (row-wise L2 norm (Q K^T)) V without explicit transposes
void run_unfused_flashsign(const fp16* Q, const fp16* K, const fp16* V, fp16* output, int Y, int X, int D) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Step 1: Compute S = Q K^T
    fp16* S;
    cudaMalloc(&S, X * Y * sizeof(fp16));
    fp16* norms;
    cudaMalloc(&norms, Y * sizeof(fp16));
    fp16 alpha = __float2half(1.0f);
    fp16 beta = __float2half(0.0f);


    //Computing S = Q K^T which is YxX
    //We have Q^T and K^T. So transpose Q.
    printf("HGEMM 1\n");
    cublasHgemm(handle,
                CUBLAS_OP_T,    // op(A) = Q
                CUBLAS_OP_N,    // op(B) = K^T
                D, Y, X,        // m = Y, n = D, k = X
                &alpha,
                Q, D,
                K, D,
                &beta,
                S, X);
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
    printf("HGEMM 2\n");
    cublasHgemm(handle,
                CUBLAS_OP_T,    // op(A) = S^T
                CUBLAS_OP_T,    // op(B) = V^T
                Y, D, X,
                &alpha,
                S, X,
                V, D,
                &beta,
                output, D);

    // Clean up
    cudaFree(S);
    cudaFree(norms);
    cublasDestroy(handle);
}

}
