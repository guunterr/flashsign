#pragma once

#include "kernels/kernel1.cuh"
#include "kernels/kernel2.cuh"
#include "kernels/kernel3.cuh"
#include "kernels/kernel4.cuh"
#include "kernels/kernel5.cuh"
#include "kernels/kernel6.cuh"
#include "utils.cu"

void run_kernel1(int M, int N, int K, const float *A, const float *B, float *C) {
    dim3 gridDim(ceil_div(M, 32), ceil_div(N, 32));
    dim3 blockDim(32, 32);
    kernel1<<<gridDim, blockDim>>>(M, N, K, A, B, C);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in run_kernel1: %s\n", cudaGetErrorString(err));
    }
    return;
}

void run_kernel2(int M, int N, int K, const float *A, const float *B, float *C) {
    dim3 gridDim(ceil_div(M, 32), ceil_div(N, 32));
    dim3 blockDim(32 * 32);
    kernel2<32><<<gridDim, blockDim>>>(M, N, K, A, B, C);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in run_kernel2: %s\n", cudaGetErrorString(err));
    }
    return;
}

void run_kernel3(int M, int N, int K, const float *A, const float *B, float *C) {
    dim3 gridDim(ceil_div(M, 32), ceil_div(N, 32));
    dim3 blockDim(32 * 32);
    cudaFuncSetAttribute(kernel3<32>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    kernel3<32><<<gridDim, blockDim>>>(M, N, K, A, B, C);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in run_kernel3: %s\n", cudaGetErrorString(err));
    }
    return;
}

void run_kernel4(int M, int N, int K, const float *A, const float *B, float *C) {
    // BK = BN/TM = BM/TM
    const int BK = 8;
    const int TM = 8;
    const int BM = 64;
    const int BN = 64;
    dim3 gridDim(ceil_div(N, BN), ceil_div(M, BM));
    dim3 blockDim((BM * BN) / TM);
    kernel4<BM, BN, BK, TM><<<gridDim, blockDim>>>(M, N, K, A, B, C);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in run_kernel4: %s\n", cudaGetErrorString(err));
    }
    return;
}

void run_kernel5(int M, int N, int K, const float *A, const float *B, float *C) {
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    const int BM = 64;
    const int BN = 64;
    dim3 gridDim(ceil_div(N, BN), ceil_div(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    kernel5<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, A, B, C);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in run_kernel5: %s\n", cudaGetErrorString(err));
    }
    return;
}

void run_kernel6(int M, int N, int K, const float *A, const float *B, float *C) {
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
        printf("CUDA error in run_kernel6: %s\n", cudaGetErrorString(err));
    }
    return;
}

void run_kernel(int kernel_number, int M, int N, int K, const float *A, const float *B, float *C) {
    switch (kernel_number) {
        case 1:
            run_kernel1(M, N, K, A, B, C);
            break;
        case 2:
            run_kernel2(M, N, K, A, B, C);
            break;
        case 3:
            run_kernel3(M, N, K, A, B, C);
            break;
        case 4:
            run_kernel4(M, N, K, A, B, C);
            break;
        case 5:
            run_kernel5(M, N, K, A, B, C);
            break;
        case 6:
            run_kernel6(M, N, K, A, B, C);
            break;
        default:
            printf("Invalid kernel number\n");
            break;
    }
    return;
}

void test_kernel(int kernel_number, bool print = false, int N = 256) {
    float *a, *b, *c1, *c2, *d_a, *d_b, *d_c1, *d_c2;
    a = (float *)malloc(N * N * sizeof(float));
    b = (float *)malloc(N * N * sizeof(float));
    c1 = (float *)malloc(N * N * sizeof(float));
    c2 = (float *)malloc(N * N * sizeof(float));

    randomise_matrix(a, N * N);
    randomise_matrix(b, N * N);
    randomise_matrix(c1, N * N);
    memcpy(c2, c1, N * N * sizeof(float));

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
    cudaMalloc((void **)&d_a, N * N * sizeof(float));
    cudaMalloc((void **)&d_b, N * N * sizeof(float));
    cudaMalloc((void **)&d_c1, N * N * sizeof(float));
    cudaMalloc((void **)&d_c2, N * N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_a, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c1, c1, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c2, c2, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    // Run reference kernel 1 and current kernel
    run_kernel(kernel_number, N, N, N, d_a, d_b, d_c2);
    run_kernel(3, N, N, N, d_a, d_b, d_c1);
    cudaDeviceSynchronize();

    // Copy reference kernel 1 and current kernel results back to host
    cudaMemcpy(a, d_a, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_b, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(c2, d_c2, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(c1, d_c1, N * N * sizeof(float), cudaMemcpyDeviceToHost);
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
    float *a, *b, *c, *d_a, *d_b, *d_c;
    printf("Timing kernel %d with %d x %d matrices\n", kernel_number, N, N);
    printf("Warming up...\n");
    for (size_t i = 0; i < warmup; i++) {
        a = make_random_matrix(N, N);
        b = make_random_matrix(N, N);
        c = make_random_matrix(N, N);

        cudaMalloc((void **)&d_a, N * N * sizeof(float));
        cudaMalloc((void **)&d_b, N * N * sizeof(float));
        cudaMalloc((void **)&d_c, N * N * sizeof(float));
        cudaMemcpyAsync(d_a, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_b, b, N * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_c, c, N * N * sizeof(float), cudaMemcpyHostToDevice);
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

        cudaMalloc((void **)&d_a, N * N * sizeof(float));
        cudaMalloc((void **)&d_b, N * N * sizeof(float));
        cudaMalloc((void **)&d_c, N * N * sizeof(float));
        cudaMemcpyAsync(d_a, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_b, b, N * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_c, c, N * N * sizeof(float), cudaMemcpyHostToDevice);
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
        printf("%.2f ms\n", times[i]);
        average_time += times[i];
        std += times[i] * times[i];
    }
    average_time /= runs;
    std = sqrt(std / runs - average_time * average_time);
    double FLOPS = 2 * pow(N, 3) + pow(N, 2);
    printf("Kernel %d took a total of %.2f+-%.2f ms , doing %.2e FLOPS, giving %.2f GFLOPS/s\n",
           kernel_number, average_time, std, FLOPS, FLOPS / (average_time * 1e6));
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return;
}