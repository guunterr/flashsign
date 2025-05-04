#include <cublas_v2.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <mma.h>

#include <cassert>
#include <ctime>
#include <cuda/barrier>
#include <iostream>
#include <random>
#include <vector>

#include "utils.cu"

typedef __half fp16;
typedef __half2 fp162;
namespace test_tensorcore {
#define CUDACHECK(x) { cudaError_t err = x; if (err != cudaSuccess) { printf("CUDA error in %s: %s\n", __func__, cudaGetErrorString(err)); exit(-1); } }
using namespace nvcuda;
namespace cg = cooperative_groups;

int ceil_div(int a, int b) {
    return (a / b) + (a % b != 0);
}

#define flip_buffer(x) (1-x)

#define WARP_SIZE 32

template<const int size>
__device__ void async_load(fp16 *src, fp16 *dst, cg::thread_block block){
    cg::memcpy_async(block, dst, src, size * sizeof(fp16));
}

__device__ void load2(fp16* src, fp16* dst){
    dst[0] = src[0];
    dst[1] = src[1];
}

#define load8_async(src, dst) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" : : "l"(dst), "l"(src))
#define load2_async(src, dst) asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" : : "l"(dst), "l"(src))

#define load2_async_prefetch(src, dst) asm volatile("cp.async.ca.shared.global.L2::256B [%0], [%1], 4;" : : "l"(dst), "l"(src))
#define load8_async_prefetch(src, dst) asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;" : : "l"(dst), "l"(src))

#define cvta_shared_64(addr, smem_ptr) asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(addr) : "l"(smem_ptr))
#define cvta_shared_32(addr, ptr) asm volatile("cvta.to.shared.u32 %0, %1;" : "=r"(addr) : "r"(ptr))

#define commit_group() asm volatile("cp.async.commit_group;")
#define wait_all() asm volatile("cp.async.wait_all;")

#define ldmatrix1(m, src) asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(m) : "l"(src))
// #define ldmatrix4(m, src) asm volatile("ldmatrix.sync.")

template<const int BX, const int BY, const int D>
__global__ void kernel(int X, fp16* Q, fp16* K, fp16* V, fp16* O, long long int * global_timers) {
    //Algorithm -> Load Q -> Prefetch KV Row -> Load Next KV Row, Process current dot product and accumualate l2 -> Normalise O -> Output
    const int tix = threadIdx.x;
    const int warpIdx = tix / WARP_SIZE;
    const int bufferPadding = 8;
    const int D_Frag_Count = D/8;
    __shared__ fp16 KVBuffer[2][2][BX][D + bufferPadding];

    //2 elements = 1 register per 8x8 fragment, D_Frag_Count needed to span D=128, 4 rows required to fill a warp
    //64 registers just for this!
    uint32_t Q_frags[4][D_Frag_Count][1];
    
    fp16* Q_start = Q + blockIdx.x * BY * D;

    const int access_size = 8;
    const int threads_per_row = D/access_size;
    for (int Q_row = 0; Q_row < 128; Q_row+=access_size)
    {
        fp16* read_ptr = Q_start + Q_row * D + tix*access_size;
        fp16* write_ptr = &KVBuffer[0][0][Q_row + (tix/threads_per_row)][(tix % threads_per_row) * access_size];
        uint64_t write_addr;
        cvta_shared_64(write_addr, write_ptr);
        load8_async_prefetch(read_ptr,write_addr);
    }

    commit_group();
    wait_all();
    __syncthreads();

    fp16* q_frag_pointer = &KVBuffer[0][0][tix][0];
    uint64_t q_frag_addr;
    cvta_shared_64(q_frag_addr, q_frag_pointer);
    ldmatrix1(Q_frags[0][0][0], q_frag_addr);

    printf("Thread %d: %.0f %.0f\n", tix, __half2float(reinterpret_cast<half2 *>(&Q_frags[0][0][0])[0].x), __half2float(reinterpret_cast<half2 *>(&Q_frags[0][0][0])[0].y));


    // if (tix == 0 && false)
    // {
    //     for (int row = 0; row < 128; row++)
    //     {
    //         printf("Row %d: ", row);
    //         for (int col = 0; col < 128; col++)
    //         {
    //             printf("%.0f ", __half2float(KVBuffer[0][0][row][col]));
    //         }
    //         printf("\n");
    //     }
        
    // }
    
    
}


void run_kernel(int X, int Y, fp16* Q, fp16* K, fp16* V, fp16* O) {
    constexpr uint BY = 128;
    constexpr uint BX = 32;
    constexpr uint D = 128;

    dim3 gridDim(ceil_div(Y, BY));
    dim3 blockDim(BY);
    printf("Running kernel\n");
    kernel<BX, BY, D><<<1, BY>>>(X, Q, K, V, O, 0);
}
}

int main(int argc, char* argv[]) {
    //Allocate and initialise Q, K, V, O
    constexpr int D = 128;
    const int X = 32;
    const int Y = 128;
    fp16 *Q, *K, *V, *O, *d_Q, *d_K, *d_V, *d_O;
    Q = (fp16*)malloc(Y * D * sizeof(fp16));
    K = (fp16*)malloc(X * D * sizeof(fp16));
    V = (fp16*)malloc(X * D * sizeof(fp16));
    O = (fp16*)malloc(Y * D * sizeof(fp16));
    cudaMalloc((void**)&d_Q, Y * D * sizeof(fp16));
    cudaMalloc((void**)&d_K, X * D * sizeof(fp16));
    cudaMalloc((void**)&d_V, X * D * sizeof(fp16));
    cudaMalloc((void**)&d_O, Y * D * sizeof(fp16));
    initialise_matrix(Q, Y, D);
    initialise_matrix(K, X, D);
    initialise_matrix(V, X, D);
    cudaMemcpy(d_Q, Q, Y * D * sizeof(fp16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, X * D * sizeof(fp16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, X * D * sizeof(fp16), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    printf("Running kernel\n");
    test_tensorcore::run_kernel(X, Y, d_Q, d_K, d_V, d_O);
    cudaDeviceSynchronize();
    cudaMemcpy(O, d_O, Y * D * sizeof(fp16), cudaMemcpyDeviceToHost);
    return 0;

}
