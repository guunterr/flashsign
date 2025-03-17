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
// #include <kernels/kernel1.cuh>
// #include <kernels/kernel2.cuh>

typedef __nv_bfloat16 bf16;

std::default_random_engine generator = std::default_random_engine(time(0));

void randomise_matrix(bf16 *matrix, int N) {
    std::normal_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < N; i++) {
        matrix[i] = __float2bfloat16(distribution(generator));
    }
}

bool verify_matrix(bf16 *matRef, bf16 *matOut, int N) {
    double diff = 0.0;
    int i;
    for (i = 0; i < N; i++) {
        diff = std::fabs(__bfloat162float(matRef[i] - matOut[i]));
        if (diff > 0.1) {
            printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
                   __bfloat162float(matRef[i]), __bfloat162float(matOut[i]), diff, i);
            return false;
        }
    }
    return true;
}

__global__ void hello() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

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

template <const uint BLOCKSIZE>
__global__ void kernel2(int M, int N, int K, const bf16 *A, const bf16 *B, bf16 *C) {
    const uint x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const uint y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (x < M && y < N) {
        bf16 temp = 0.0;
        for (int i = 0; i < K; ++i) {
            temp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = temp + C[x * N + y];
    }
}

template <const int BLOCKSIZE>
__global__ void kernel3(int M, int N, int K, const bf16 *A, const bf16 *B, bf16 *C) {
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    __shared__ bf16 As[BLOCKSIZE * BLOCKSIZE];
    __shared__ bf16 Bs[BLOCKSIZE * BLOCKSIZE];
    // Initial locations of A, B, C blocks
    int Ablock = cRow * K * BLOCKSIZE;
    int Bblock = cCol * BLOCKSIZE;
    int Cblock = cRow * N * BLOCKSIZE + cCol * BLOCKSIZE;
    bf16 temp = 0.0;
    // Threadcol consecutive threads -> threads in warp access same elements of A, consecutive elements of B
    const uint threadCol = threadIdx.x % BLOCKSIZE;
    const uint threadRow = threadIdx.x / BLOCKSIZE;
    for (int block = 0; block < K; block += BLOCKSIZE) {
        // Load A and B blocks into shared memory
        As[threadRow * BLOCKSIZE + threadCol] = A[Ablock + threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[Bblock + threadRow * K + threadCol];
        // Wait for all threads to finish loading
        __syncthreads();
        // Move a block right along A, down along B
        Ablock += BLOCKSIZE;
        Bblock += BLOCKSIZE * N;
        // One value of C per thread = dot product of row of A and column of B
        for (int i = 0; i < BLOCKSIZE; i++) {
            temp += As[threadRow * BLOCKSIZE + i] * Bs[i * BLOCKSIZE + threadCol];
        }
        // Have to wait for all threads to finish computing before moving on
        __syncthreads();
    }
    // Write result to C
    C[Cblock + threadRow * N + threadCol] += temp;
}

int ceil_div(int a, int b) {
    return (a / b) + (a % b != 0);
}

void run_kernel1(int M, int N, int K, const bf16 *A, const bf16 *B, bf16 *C) {
    dim3 gridDim(ceil_div(M, 32), ceil_div(N, 32));
    dim3 blockDim(32, 32);
    kernel1<<<gridDim, blockDim>>>(M, N, K, A, B, C);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in run_kernel1: %s\n", cudaGetErrorString(err));
    }
    return;
}

void run_kernel2(int M, int N, int K, const bf16 *A, const bf16 *B, bf16 *C) {
    dim3 gridDim(ceil_div(M, 32), ceil_div(N, 32));
    dim3 blockDim(32 * 32);
    kernel2<32><<<gridDim, blockDim>>>(M, N, K, A, B, C);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in run_kernel2: %s\n", cudaGetErrorString(err));
    }
    return;
}

void run_kernel3(int M, int N, int K, const bf16 *A, const bf16 *B, bf16 *C) {
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

void run_kernel(int kernel_number, int M, int N, int K, const bf16 *A, const bf16 *B, bf16 *C) {
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
        default:
            printf("Invalid kernel number\n");
            break;
    }
    return;
}

bf16 *make_random_matrix(int M, int N) {
    bf16 *matrix = (bf16 *)malloc(M * N * sizeof(bf16));
    randomise_matrix(matrix, M * N);
    return matrix;
}

void warmup_kernel() {
    return;
}

void time_kernel(int kernel_number, int N = 1 << 12, int warmup = 2, int runs = 5) {
    bf16 *a, *b, *c, *d_a, *d_b, *d_c;
    printf("Timing kernel %d with %d x %d matrices\n", kernel_number, N, N);
    printf("Warming up...\n");
    for (size_t i = 0; i < warmup; i++)
    {
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
    for (size_t i = 0; i < runs; i++)
    {
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
    for (size_t i = 0; i < runs; i++)
    {
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

void print_matrix(bf16 *matrix, int M, int N) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            printf("%6.2f ", __bfloat162float(matrix[i * N + j]));
        }
        printf("\n");
    }
    return;
}

void test_kernel(int kernel_number, bool print = false, int N = 1 << 8) {
    bf16 *a, *b, *c1, *c2, *d_a, *d_b, *d_c1, *d_c2;
    a = (bf16 *)malloc(N * N * sizeof(bf16));
    b = (bf16 *)malloc(N * N * sizeof(bf16));
    c1 = (bf16 *)malloc(N * N * sizeof(bf16));
    c2 = (bf16 *)malloc(N * N * sizeof(bf16));

    randomise_matrix(a, N * N);
    randomise_matrix(b, N * N);
    randomise_matrix(c1, N * N);
    memcpy(c2, c1, N * N * sizeof(bf16));

    if (print) {
        printf("A: %dx%d\n", N, N);
        print_matrix(a, N, N);
        printf("B: %dx%d\n", N, N);
        print_matrix(b, N, N);
        printf("C: %dx%d\n", N, N);
        print_matrix(c1, N, N);
        printf("C: %dx%d\n", N, N);
        print_matrix(c2, N, N);
    }

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
    run_kernel(1, N, N, N, d_a, d_b, d_c1);
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

int main(void) {
    time_kernel(2, 1024);
    time_kernel(3, 1024);
    // time_kernel(2, 4096);
    // time_kernel(3, 4096);

    return 0;
}