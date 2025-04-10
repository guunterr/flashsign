#pragma once

#include "kernels/flashsign_0_unfused.cuh"
#include "kernels/flashsign_1_basic.cuh"
#include "kernels/flashsign_2_half2.cuh"
#include "utils.cu"
#include <cublas_v2.h>

template<const int X, const int D>
void run_flashsign(int kernel_number, int Y, fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    using flashsign_1_basic::run_flashsign1;
    using flashsign_0_unfused::run_unfused_flashsign;
    using flashsign_2_half2::run_flashsign2_half;
    switch(kernel_number) {
        case 0:
            run_unfused_flashsign(Q, K, V, O, Y, X, D);
            break;
        case 1:
            run_flashsign1<X,D>(Y, Q, K, V, O);
            break;
        case 2:
            run_flashsign2_half<X,D>(Y, Q, K, V, O);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in run_flashsign kernel %d: %s\n", kernel_number, cudaGetErrorString(err));
    }
    return;
}

template<const int X, const int D>
void test_flashsign(int kernel_number, int Y, fp16 epsilon = 0.01){
    fp16 *Q, *K, *V, *O1, *O2, *d_Q, *d_K, *d_V, *d_O1, *d_O2;
    printf("Filling out data: Q[%d x %d], K[%d x %d], V[%d x %d], Y[%d x %d]\n", Y, D, X, D, X, D, Y, D);
    Q = (fp16*)malloc(Y * D * sizeof(fp16));
    K = (fp16*)malloc(X * D * sizeof(fp16));
    V = (fp16*)malloc(X * D * sizeof(fp16));
    O1 = (fp16*)malloc(Y * D * sizeof(fp16));
    O2 = (fp16*)malloc(Y * D * sizeof(fp16));
    cudaMalloc((void**)&d_Q, Y * D * sizeof(fp16));
    cudaMalloc((void**)&d_K, X * D * sizeof(fp16));
    cudaMalloc((void**)&d_V, X * D * sizeof(fp16));
    cudaMalloc((void**)&d_O1, Y * D * sizeof(fp16));
    cudaMalloc((void**)&d_O2, Y * D * sizeof(fp16));
    cudaDeviceSynchronize();
    randomise_matrix(Q, Y*D);
    randomise_matrix(K, X*D);
    randomise_matrix(V, X*D);
    printf("%f, %f, %f", fp162f(Q[17]), fp162f(K[17]), fp162f(V[17]));
    printf("Moving data\n");
    cudaMemcpy(d_Q, Q, Y * D * sizeof(fp16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, X * D * sizeof(fp16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, X * D * sizeof(fp16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_O1, O1, Y * D * sizeof(fp16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_O2, O2, Y * D * sizeof(fp16), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    printf("Running reference kernel\n");
    run_flashsign<X,D>(0, Y, d_Q, d_K, d_V, d_O1);
    cudaDeviceSynchronize();
    printf("Running kernel %d\n", kernel_number);
    run_flashsign<X,D>(kernel_number, Y, d_Q, d_V, d_K, d_O2);
    cudaDeviceSynchronize();
    printf("Copying data back\n");
    cudaMemcpy(O1, d_O1, Y * D * sizeof(fp16), cudaMemcpyDeviceToHost);
    cudaMemcpy(O2, d_O2, Y * D * sizeof(fp16), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    verify_matrix(O1, O2, Y, D, epsilon);
    printf("%f, %f", fp162f(O1[17]), fp162f(O2[17]));
    free(Q);
    free(K);
    free(V);
    free(O1);
    free(O2);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O1);
    cudaFree(d_O2);
}

template<const int X, const int D>
void time_flashsign(int kernel_number, int Y, int warmup = 2, int runs = 5) {
    int data_len = warmup + runs;
    fp16 *Q, *K, *V, *O, *d_Q, *d_K, *d_V, *d_O;
    printf("Filling out data: Q[%d x %d], K[%d x %d], V[%d x %d], Y[%d x %d]\n", Y, D, X, D, X, D, Y, D);
    Q = (fp16*)malloc(data_len * Y * D * sizeof(fp16));
    K = (fp16*)malloc(data_len * X * D * sizeof(fp16));
    V = (fp16*)malloc(data_len * X * D * sizeof(fp16));
    O = (fp16*)malloc(data_len * Y * D * sizeof(fp16));
    cudaMalloc((void**)&d_Q, data_len * Y * D * sizeof(fp16));
    cudaMalloc((void**)&d_K, data_len * X * D * sizeof(fp16));
    cudaMalloc((void**)&d_V, data_len * X * D * sizeof(fp16));
    cudaMalloc((void**)&d_O, data_len * Y * D * sizeof(fp16));
    for (int i = 0; i < data_len; i++)
    {
        randomise_matrix(Q + i * Y * D, Y * D);
        randomise_matrix(K + i * X * D, X * D);
        randomise_matrix(V + i * X * D, X * D);
    }
    cudaMemcpyAsync(d_Q, Q, data_len * Y * D * sizeof(fp16), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_K, K, data_len * X * D * sizeof(fp16), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_V, V, data_len * X * D * sizeof(fp16), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_O, O, data_len * Y * D * sizeof(fp16), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    printf("Timing kernel...\n");
    float times[runs];
    for (int i = 0; i < data_len; i++){
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        // run_flashsign<X,D>(kernel_number, Y, d_Q + i * Y * D, d_K + i * X * D, d_V  + i * X * D, d_O + i * Y * D);
        run_flashsign<X,D>(kernel_number, Y, d_Q, d_K, d_V, d_O);
        cudaEventRecord(stop);
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);

        cudaDeviceSynchronize();
        if(i < warmup) continue;
        cudaEventElapsedTime(&times[i - warmup], start, stop);
    }

    float average_time = 0;
    float std = 0;
    printf("Kernel %d took the following times:\n", kernel_number);
    for (size_t i = 0; i < runs; i++) {
        printf("%.4f ms, ", times[i]);
        if(i % 10 == 9) printf("\n");
        average_time += times[i];
        std += times[i] * times[i];
    }
    average_time /= runs;
    std = sqrt(std / runs - average_time * average_time);
    float relative_error = std / average_time;
    printf("\nKernel %d took a total of %.4f+-%.4f ms\n", kernel_number, average_time, std);
    free(Q);
    free(K);
    free(V);
    free(O);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
}