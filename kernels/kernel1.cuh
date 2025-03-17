#pragma once

__global__ void kernel1(int M, int N, int K, const bf16 *A, const bf16 *B, bf16 *C) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        
        bf16 temp = 0.0;
        for (int i = 0; i < K; ++i) {
            temp += A[x * K + i] * B[i * N + y];
        }
        bf16 new_val = temp + C[x * N + y];
        C[x * N + y] = new_val;
    }
}