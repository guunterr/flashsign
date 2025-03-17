#pragma once
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

typedef __nv_bfloat16 bf16;

// https://godbolt.org/z/q6Grbv5Ps
// This is apparently supposed to tell me something!?!?! Performance is magic.
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