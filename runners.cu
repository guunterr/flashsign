#pragma once

#include "kernels/kernel_1_basic.cuh"
#include "utils.cu"
#include <cublas_v2.h>

cublasHandle_t handle;
void run_kernel_basic(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    const int BM = 64;
    const int BN = 64;
    dim3 gridDim(ceil_div(N, BN), ceil_div(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    kernel6<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, A, B, C);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in run_kernel_basic: %s\n", cudaGetErrorString(err));
    }
    return;
}

void run_kernel_cublas(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C){

    const float alpha = 1.0f;
    const float beta = 1.0f;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16BF, N, A, CUDA_R_16BF, K, &beta, C, CUDA_R_16BF, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in run_kernel_cublas: %s\n", cudaGetErrorString(err));
    }
    return;
}

void run_kernel(int kernel_number, int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
    switch (kernel_number) {
        case 0:
            run_kernel_cublas(M, N, K, A, B, C);
            break;
        case 1:
            run_kernel_basic(M, N, K, A, B, C);
            break;
        default:
            printf("Invalid kernel number\n");
            break;
    }
    return;
}

void test_kernel(int kernel_number, bool print = false, int N = 256) {
    bf16 *a, *b, *c1, *c2, *d_a, *d_b, *d_c1, *d_c2;
    a = (bf16 *)malloc(N * N * sizeof(bf16));
    b = (bf16 *)malloc(N * N * sizeof(bf16));
    c1 = (bf16 *)malloc(N * N * sizeof(bf16));
    c2 = (bf16 *)malloc(N * N * sizeof(bf16));

    randomise_matrix(a, N * N);
    randomise_matrix(b, N * N);
    randomise_matrix(c1, N * N);
    memcpy(c2, c1, N * N * sizeof(bf16));

    // if (print) {
    //     printf("A: %dx%d\n", N, N);
    //     print_matrix(a, N, N);
    //     printf("B: %dx%d\n", N, N);
    //     print_matrix(b, N, N);
    //     printf("C: %dx%d\n", N, N);
    //     print_matrix(c1, N, N);
    //     printf("C: %dx%d\n", N, N);
    //     print_matrix(c2, N, N);
    // }

    // Allocate memory on device
    cudaMalloc((void **)&d_a, N * N * sizeof(bf16));
    cudaMalloc((void **)&d_b, N * N * sizeof(bf16));
    cudaMalloc((void **)&d_c1, N * N * sizeof(bf16));
    cudaMalloc((void **)&d_c2, N * N * sizeof(bf16));

    // Copy data to device
    cudaMemcpy(d_a, a, N * N * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * N * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c1, c1, N * N * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c2, c2, N * N * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    // Run reference kernel 1 and current kernel
    run_kernel(kernel_number, N, N, N, d_a, d_b, d_c2);
    run_kernel(0, N, N, N, d_a, d_b, d_c1);
    cudaDeviceSynchronize();

    // Copy reference kernel 1 and current kernel results back to host
    cudaMemcpy(a, d_a, N * N * sizeof(bf16), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_b, N * N * sizeof(bf16), cudaMemcpyDeviceToHost);
    cudaMemcpy(c2, d_c2, N * N * sizeof(bf16), cudaMemcpyDeviceToHost);
    cudaMemcpy(c1, d_c1, N * N * sizeof(bf16), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    bool pass = verify_matrix(c2, c1, N * N);
    printf("Kernel %d: %s\n", kernel_number, pass ? "PASS" : "FAIL");
    if (print && !pass) {
        printf("A: %dx%d\n", N, N);
        print_matrix(a, N, N);
        printf("B: %dx%d\n", N, N);
        print_matrix(b, N, N);
        printf("Kernel %d result:\n", kernel_number);
        print_matrix(c1, N, N);
        printf("Reference result:\n");
        print_matrix(c2, N, N);
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c1);
    cudaFree(d_c2);
    free(a);
    free(b);
    free(c1);
    free(c2);
    return;
}

void time_kernel(int kernel_number, int N = 1 << 12, int warmup = 2, int runs = 5) {
    bf16 *a, *b, *c, *d_a, *d_b, *d_c;
    printf("Timing kernel %d with %d x %d matrices\n", kernel_number, N, N);
    printf("Warming up...\n");
    for (size_t i = 0; i < warmup; i++) {
        a = make_random_matrix(N, N);
        b = make_random_matrix(N, N);
        c = make_random_matrix(N, N);

        cudaMalloc((void **)&d_a, N * N * sizeof(bf16));
        cudaMalloc((void **)&d_b, N * N * sizeof(bf16));
        cudaMalloc((void **)&d_c, N * N * sizeof(bf16));
        cudaMemcpyAsync(d_a, a, N * N * sizeof(bf16), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_b, b, N * N * sizeof(bf16), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_c, c, N * N * sizeof(bf16), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        run_kernel(kernel_number, N, N, N, d_a, d_b, d_c);

        cudaDeviceSynchronize();
    }
    printf("Timing kernel...\n");
    float times[runs];
    for (size_t i = 0; i < runs; i++) {
        // Initialise and copy matrices
        a = make_random_matrix(N, N);
        b = make_random_matrix(N, N);
        c = make_random_matrix(N, N);

        cudaMalloc((void **)&d_a, N * N * sizeof(bf16));
        cudaMalloc((void **)&d_b, N * N * sizeof(bf16));
        cudaMalloc((void **)&d_c, N * N * sizeof(bf16));
        cudaMemcpyAsync(d_a, a, N * N * sizeof(bf16), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_b, b, N * N * sizeof(bf16), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_c, c, N * N * sizeof(bf16), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        // Run and time kernel
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        run_kernel(kernel_number, N, N, N, d_a, d_b, d_c);
        cudaEventRecord(stop);
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);
        cudaDeviceSynchronize();

        cudaEventElapsedTime(&times[i], start, stop);
    }
    float average_time = 0;
    float std = 0;
    printf("Kernel %d took the following times:\n", kernel_number);
    for (size_t i = 0; i < runs; i++) {
        printf("%.4f ms\n", times[i]);
        average_time += times[i];
        std += times[i] * times[i];
    }
    average_time /= runs;
    std = sqrt(std / runs - average_time * average_time);
    float relative_error = std / average_time;
    double FLOPS = 2 * pow(N, 3) + pow(N, 2);
    printf("Kernel %d took a total of %.4f+-%.4f ms , doing %.2e FLOPS, giving %.2f +- %.2f TFLOPS/s\n",
           kernel_number, average_time, std, FLOPS, FLOPS / (average_time * 1e9), (FLOPS / (average_time * 1e9)) * relative_error);
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return;
}