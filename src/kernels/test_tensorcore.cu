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

namespace tensorcore_kernel {
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

#define flip(x) (1-x)

#define assertions() \
    assert(BX % 8 == 0);\
    assert(BX >= 8);\
    assert(D % 16 == 0);\
    assert(BX == 8)

template<const int size>
__device__ void async_load(fp16 *src, fp16 *dst, cg::thread_block block){
    cg::memcpy_async(block, dst, src, size * sizeof(fp16));
}

template<const int BX, const int BY, const int D>
__global__ void kernel(int X, fp16* Q, fp16* K, fp16* S) {
    //Algorithm -> Load Q -> Prefetch KV Row -> Load Next KV Row, Process current dot product and accumualate l2 -> Normalise O -> Output
    assertions();
    constexpr Q_Frag_Count = D/16;
    wmma::fragment<wmma::matrix_a, 32, 8, 16, half, wmma::row_major> Q_Frag[Q_Frag_Count];
    wmma::fragment<wmma::matrix_b, 32, 8, 16, half, wmma::col_major> K_Frag;
    wmma::fragment<wmma::accumulator, 32, 8, 16, half> S_Frag;

    wmma::fill_fragment(S_Frag, 0);

    __shared__ fp16 KVBuffer[2][2][BX * D];

    Q += blockIdx.x * BY * D;
    O += blockIdx.x * BY * D;

    const int tix = threadIdx.x
    cg::thread_block block = cg::this_thread_block();

    for (int QLoadChunk = 0; QLoadChunk < BY; QLoadChunk += 4 * BX)
    {
        async_load<4 * BX * D>(Q + QLoadChunk * D, &KVBuffer[0][0][0], block);
        cg::wait(block);
        const int rowIdx = (tix - QLoadChunk);
        if(rowIdx >= 0 && rowIdx <= 4 * BX){
            for (int frag_idx = 0; frag_idx < Q_Frag_Count; frag_idx++)
            {
                wmma::load_matrix_sync(Q_Frag[frag_idx], &KVBuffer[0][0][0] + (rowIdx / warpSize) * warpSize * D + frag_idx * 16, D);
            }
        }
    }

    int active_buffer = 0;
    //Prefetch first KV Buffer
    async_load<BX * D>(K, &KVBuffer[active_buffer][0][0], block);
    async_load<BX * D>(V, &KVBuffer[active_buffer][1][0], block);

    for (int KVLoadChunk = 0; KVLoadChunk < X/BX; KVLoadChunk += 1)
    {
        //Wait until buffer is filled
        cg::wait(block);
        //Prefetch next KV Buffer
        async_load<BX * D>(K + (KVLoadChunk + 1) * BX * D, &KVBuffer[flip(active_buffer)][0][0], block);
        async_load<BX * D>(V + (KVLoadChunk + 1) * BX * D, &KVBuffer[flip(active_buffer)][1][0], block);
        //Process filled buffer

        //Compute QK^T dot product
        for (int Kfragidx = 0; Kfragidx < count; Kfragidx++)
        {
            wmma::load_matrix_sync(K_Frag, &KVBuffer[active_buffer][0][0] + Kfragidx * 16, D);
            for (int mul_idx = 0; mul_idx < Q_Frag_Count; mul_idx++)
            {
                wmma::mma_sync(S_Frag, Q_Frag[mul_idx], K_Frag, S_Frag);
            }
            
        }
        

        //Calculate L2 norm of S

        //Compute V
    }
    
    
}
}