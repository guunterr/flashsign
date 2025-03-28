#pragma once

std::default_random_engine generator = std::default_random_engine(time(0));

void randomise_matrix(float *matrix, int N) {
    std::normal_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < N; i++) {
        matrix[i] = distribution(generator);
    }
}

float *make_random_matrix(int M, int N) {
    float *matrix = (float *)malloc(M * N * sizeof(float));
    randomise_matrix(matrix, M * N);
    return matrix;
}

bool verify_matrix(float *matRef, float *matOut, int N) {
    double diff = 0.0;
    int i;
    for (i = 0; i < N; i++) {
        diff = std::fabs(matRef[i] - matOut[i]);
        if (diff > 0.1) {
            printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
                   matRef[i], matOut[i], diff, i);
            return false;
        }
    }
    return true;
}
int ceil_div(int a, int b) {
    return (a / b) + (a % b != 0);
}


void print_matrix(float *matrix, int M, int N) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            printf("%6.2f ", (matrix[i * N + j]));
        }
        printf("\n");
    }
    return;
}


void get_device_properties() {
    int device;
    cudaGetDevice(&device);

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("Device Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Threads/Block: %d\n", prop.maxThreadsPerBlock);
    printf("Threads/SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Threads/Warp: %d\n", prop.warpSize);
    printf("Regs/Block: %d\n", prop.regsPerBlock);
    printf("Regs/SM: %d\n", prop.regsPerMultiprocessor);
    printf("Total Global Mem: %zu MB\n", prop.totalGlobalMem / 1024 / 1024);
    printf("Shared Mem per Block: %zu KB \n", prop.sharedMemPerBlock / 1024);
    printf("Smem Overhead from CUDA Runtime: %zu bytes\n", prop.reservedSharedMemPerBlock);
    printf("Smem per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
    printf("SM Count: %d\n", prop.multiProcessorCount);
    printf("Max Warps per SM: %d\n", prop.maxThreadsPerMultiProcessor / prop.warpSize);
}