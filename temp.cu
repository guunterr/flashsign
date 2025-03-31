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

int main(int argc, char* argv[]) {
    get_device_properties();
    int kernel_number = atoi(argv[2]);
    int warmup = atoi(argv[3]);
    int runs = atoi(argv[4]);
    if (kernel_number == 0 || runs == 0) {
        printf("Invalid arguments\n");
    }
    if (argv[1][0] == 'r') {
        printf("Running kernel %d for %d warmup and %d runs\n", kernel_number, warmup, runs);
        time_kernel(kernel_number, 4096, warmup, runs);
    } else {
        printf("Testing kernel %d\n", kernel_number);
        test_kernel(kernel_number, false);
    }

    return 0;
}