#include <cmath>
#include <iostream>
#include <cublas_v2.h>
#include "utils.cu"
// Kernel function to add the elements of two arrays

__global__ void hello() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

__global__ void add(float *a, float *b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = a[i] + b[i];
    }
}

__global__ void sgemm_kernel(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
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
    float *a, *b, *d_a, *d_b;
    int N = 1 << 12;
    a = (float *)malloc(N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));

    randomise_matrix(a, N);
    randomise_matrix(b, N);
    cudaMalloc((void **)&d_a, N * sizeof(float));
    cudaMalloc((void **)&d_b, N * sizeof(float));
    cudaMemcpyAsync(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsnyc(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    add<<<1, 1>>>(d_a, d_b, N);
    cudaDeviceSynchronize();
    cudaMemcpy(a, d_a, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    free(a);
    free(b);

    return 0;
}