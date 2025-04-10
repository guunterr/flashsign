#include <cublas_v2.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
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

#include "runners.cu"
#include "utils.cu"

typedef __half fp16;

int main(int argc, char* argv[]) {
    clock_t start = clock();
    int kernel_number = atoi(argv[2]);
    int warmup = atoi(argv[3]);
    int runs = atoi(argv[4]);
    fp16 epsilon = __double2half(atof(argv[5]));
    if (kernel_number == 0 || runs == 0) {
        printf("Invalid arguments\n");
    }
    if (argv[1][0] == 'r') {
        printf("Running flashsign_kernel %d for %d warmup and %d runs\n", kernel_number, warmup, runs);
        time_flashsign<16384, 128>(kernel_number, 16384, warmup, runs);
    } else{
        printf("Testing flashsign kernel %d\n", kernel_number);
        test_flashsign<1024, 128>(kernel_number, 1024, epsilon);
    }
    clock_t end = clock();
    double duration = (end-start) / CLOCKS_PER_SEC;
    printf("Run wallclock time: %.3f s\n", duration);

    return 0;
}