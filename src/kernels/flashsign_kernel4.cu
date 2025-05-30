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

namespace flashsign_kernel4 {
#define CUDACHECK(x) { cudaError_t err = x; if (err != cudaSuccess) { printf("CUDA error in %s: %s\n", __func__, cudaGetErrorString(err)); exit(-1); } }
using namespace nvcuda;
namespace cg = cooperative_groups;

typedef __half fp16;
typedef __half2 fp162;

int ceil_div(int a, int b) {
    return (a / b) + (a % b != 0);
}

//BX multiple of 8, bigger than 8 -> 4 * BX is a full warp's worth of threads
//BX must actually be equal to 8 for now


#define TIMER_METRICS 4
#define init_timers() \
    long long int starts[TIMER_METRICS] = {};\
    long long int timers[TIMER_METRICS] = {}

#define start_timer(i) \
    if (tix % WARP_SIZE == 0) starts[i] = clock64()

#define end_timer(i) \
    if (tix % WARP_SIZE == 0) timers[i] += clock64() - starts[i]


#define flip(x) (1-x)

#define assertions() \
    assert(BX % 8 == 0);\
    assert(BX >= 8);\
    assert(D % 16 == 0);\
    assert(BX == 8)

#define WARP_SIZE 32

template<const int size>
__device__ void async_load(fp16 *src, fp16 *dst, cg::thread_block block){
    cg::memcpy_async(block, dst, src, size * sizeof(fp16));
}

template<const int BX, const int BY, const int D>
__global__ void kernel(int X, fp16* Q, fp16* K, fp16* V, fp16* O, long long int * global_timers) {
    //Algorithm -> Load Q -> Prefetch KV Row -> Load Next KV Row, Process current dot product and accumualate l2 -> Normalise O -> Output
    const int tix = threadIdx.x;
    const int warpIdx = tix / WARP_SIZE;
    constexpr int Q_Frag_Count = D/16;
    assertions();
    init_timers();
    wmma::fragment<wmma::matrix_a, 32, 8, 16, half, wmma::row_major> Q_Frag[Q_Frag_Count];
    wmma::fragment<wmma::matrix_b, 32, 8, 16, half, wmma::col_major> K_Frag;
    wmma::fragment<wmma::accumulator, 32, 8, 16, half, void> S_Frag;
    wmma::fragment<wmma::matrix_b, 32, 8, 16, half, wmma::row_major> V_Frag;

    wmma::fill_fragment(S_Frag, CUDART_ZERO_FP16);

    __shared__ fp16 KVBuffer[2][2][BX * D];

    Q += blockIdx.x * BY * D;
    O += blockIdx.x * BY * D;

    
    cg::thread_block block = cg::this_thread_block();
    start_timer(0);
    for (int QLoadChunk = 0; QLoadChunk < BY; QLoadChunk += 4 * BX)
    {
        
        async_load<4 * BX * D>(Q + QLoadChunk * D, &KVBuffer[0][0][0], block);
        cg::wait(block);
        const int rowIdx = (tix - QLoadChunk);
        if(rowIdx >= 0 && rowIdx < 4 * BX){
            // printf("Warp %d thread %d, threadIdx %d in condition\n", tix / warpSize, tix % warpSize, tix);
            for (int frag_idx = 0; frag_idx < Q_Frag_Count; frag_idx++)
            {
                wmma::load_matrix_sync(Q_Frag[frag_idx], &KVBuffer[0][0][0] + (rowIdx / WARP_SIZE) * WARP_SIZE * D + frag_idx * 16, D);
            }
        }
        
    }
    end_timer(0);

    int active_buffer = 0;
    //Prefetch first KV Buffer
    async_load<BX * D>(K, &KVBuffer[active_buffer][0][0], block);
    async_load<BX * D>(V, &KVBuffer[active_buffer][1][0], block);

    start_timer(1);
    for (int KVLoadChunk = 0; KVLoadChunk < X/BX; KVLoadChunk += 1)
    {
        start_timer(2);
        //Wait until buffer is filled
        cg::wait(block);
        //Prefetch next KV Buffer
        async_load<BX * D>(K + (KVLoadChunk + 1) * BX * D, &KVBuffer[flip(active_buffer)][0][0], block);
        async_load<BX * D>(V + (KVLoadChunk + 1) * BX * D, &KVBuffer[flip(active_buffer)][1][0], block);
        end_timer(2);
        //Process filled buffer
        start_timer(3);
        // Compute QK^T dot product
        for (int Kfragidx = 0; Kfragidx < D/16; Kfragidx++)
        {
            wmma::load_matrix_sync(K_Frag, &KVBuffer[active_buffer][0][0] + Kfragidx * 16, D);
            for (int mul_idx = 0; mul_idx < Q_Frag_Count; mul_idx++)
            {
                wmma::mma_sync(S_Frag, Q_Frag[mul_idx], K_Frag, S_Frag);
            }
            
        }
        end_timer(3);

        //Calculate L2 norm of S

        //Compute V

        //Flip buffer
        active_buffer = flip(active_buffer);
    }
    end_timer(1);
    if(tix % WARP_SIZE == 0){
        for (int i = 0; i < TIMER_METRICS; i++)
        {
            global_timers[blockIdx.x * (blockDim.x / WARP_SIZE) * TIMER_METRICS + warpIdx * TIMER_METRICS + i] = timers[i];
        }
    }
}


void run_kernel(int X, int Y, fp16* Q, fp16* K, fp16* V, fp16* O) {
    long long int * times, * d_times;
    long long int summarised_times[TIMER_METRICS] = {};
    constexpr uint BY = 128;
    constexpr uint BX = 8;
    constexpr uint D = 128;
    int blockCount = ceil_div(Y, BY);
    int warpCount = BY / WARP_SIZE;
    int timer_count = TIMER_METRICS * blockCount * warpCount;

    cudaMalloc((void**)&d_times, sizeof(long long int) * TIMER_METRICS * blockCount * warpCount);
    times = (long long int *) malloc(sizeof(long long int) * TIMER_METRICS * blockCount * warpCount);
    cudaDeviceSynchronize();
    cudaMemset(d_times, 0, sizeof(long long int) * TIMER_METRICS * blockCount * warpCount);
    cudaDeviceSynchronize();
    dim3 gridDim(ceil_div(Y, BY));
    dim3 blockDim(BY);
    kernel<BX, BY, D><<<gridDim, blockDim>>>(X, Q, K, V, O, d_times);
    cudaDeviceSynchronize();
    cudaMemcpy(times, d_times, sizeof(long long int) * TIMER_METRICS * blockCount * warpCount, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int metricIdx = 0; metricIdx < TIMER_METRICS; metricIdx++)
    {
        for (int block_counter = 0; block_counter < blockCount; block_counter++)
        {
            for (int warp_counter = 0; warp_counter < warpCount; warp_counter++)
            {
                long long int metric = times[block_counter * TIMER_METRICS * warpCount + warp_counter * TIMER_METRICS + metricIdx];
                summarised_times[metricIdx] += metric;
            }
        }
        summarised_times[metricIdx] /= blockCount * warpCount;
        
    }
    printf("Flashsign Kernel 4 Metrics: QLoad = %lld, Total_streamtime = %lld, KVLoadWait = %lld, DotTime = %lld\n", \
        summarised_times[0], summarised_times[1], summarised_times[2], summarised_times[3]);
    
}
}

#include <torch/types.h>
#include <ATen/ATen.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor flashsign_4(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);
    //change
    int Y = Q.size(0);
    int X = K.size(0);
    const int D = 128;
    
    assert(D % 2 == 0);
    
    assert(K.size(1) == D);
    assert(V.size(1) == D);
    assert(Q.size(1) == D);

    assert(V.size(0) == X);


    // Create matrices according to the number of splits.
    auto O = torch::zeros({Y, D}, Q.options());
    // Set up the pointers.
    __half* Qp = (__half*)Q.data_ptr<at::Half>();
    __half* Kp = (__half*)K.data_ptr<at::Half>();
    __half* Vp = (__half*)V.data_ptr<at::Half>();
    __half* Op = (__half*)O.data_ptr<at::Half>();
    
    flashsign_kernel4::run_kernel(X, Y, Qp, Kp, Vp, Op);

    return O;
}