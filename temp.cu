#include "utils.cu"
#include <stdio.h>
#include <cuda_runtime.h>
// Kernel function to add the elements of two arrays

__global__ void hello() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}


__global__ void kernel1(const int M, const int N, const int K, const bf16 alpha, const bf16 *A, const bf16 *B, bf16 beta, bf16 *C) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float temp = 0.0;
        for (int i = 0; i < K; ++i) {
            temp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * temp + beta * C[x * N + y];
    }
}

int main(void) {
    bf16 *a, *b, *d_a, *d_b;
    int N = 1 << 12;
    a = (bf16 *)malloc(N * sizeof(bf16));
    b = (bf16 *)malloc(N * sizeof(bf16));

    randomise_matrix(a, N);
    randomise_matrix(b, N);
    cudaMalloc((void **)&d_a, N * sizeof(bf16));
    cudaMalloc((void **)&d_b, N * sizeof(bf16));
    cudaMemcpyAsync(d_a, a, N * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_b, b, N * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaDeviceSynchronize();
    cudaMemcpy(a, d_a, N * sizeof(bf16), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    free(a);
    free(b);

    return 0;
}