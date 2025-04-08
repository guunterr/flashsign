#pragma once

#include "kernels/flashsign_1_basic.cuh"
#include "utils.cu"
#include <cublas_v2.h>

template<const int X, const int D>
void run_flashsign(int kernel_number, int Y, fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    using flashsign_1_basic::run_flashsign1;
    switch(kernel_number) {
        case 1:
            run_flashsign1<X,D>(Y, Q, K, V, O);
            break;
    }
    return;
}
template<const int X, const int D>
void time_flashsign(int kernel_number, int Y, int warmup = 2, int runs = 5) {
    int data_len = warmup + runs;
    fp16 *Q, *K, *V, *O, *d_Q, *d_K, *d_V, *d_O;
    printf("Filling out data: Q[%d x %d], K[%d x %d], V[%d x %d], Y[%d x %d]\n", Y, D, X, D, X, D, X, D);
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
        randomise_matrix(O + i * Y * D, Y * D);
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
    printf("Kernel %d took a total of %.4f+-%.4f ms", kernel_number, average_time, std);
    free(Q);
    free(K);
    free(V);
    free(O);
}