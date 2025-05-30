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

#include <torch/types.h>
#include <ATen/ATen.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)



namespace flashsign_kernel3 {
using namespace nvcuda;
namespace cg = cooperative_groups;

typedef __half fp16;
typedef __half2 fp162;

int ceil_div(int a, int b) {
    return (a / b) + (a % b != 0);
}


//64 registers for Q
//64 registers for O
//4 registers for S
//1 register for l
//4 registers for temporary work

template<const int NUM_THREADS, const int size>
__device__ void loadGMEMToSMEM(fp162 *src, fp162 *dst, cg::thread_block& group){
    cg::memcpy_async(group, dst, src, size * sizeof(fp162));
}

template<const int NUM_THREADS, const int size>
__device__ void storeSMEMtoGMEM(fp162 *src, fp162 *dst, cg::thread_block block){
    for (int i = 0; i < size; i+= 4 * NUM_THREADS)
    {
        float4 tmp = reinterpret_cast<float4 *>(&src[i + 4 * threadIdx.x])[0];
        reinterpret_cast<float4 *>(&dst[i + 4 * threadIdx.x])[0] = tmp;
    }
}

__device__ void sync(cg::thread_block& group) {
    cg::wait(group);
}

//64 registers for Q
template<const int BX, const int BY, const int D>
__global__ void kernel(int X, fp162 *Q, fp162 *K, fp162 *V, fp162 *O, long long int *times, bool time = false) {
    cg::thread_block block = cg::this_thread_block();
    //First index is double buffering, second is K vs V
    //[First K buffer][First V Buffer][Second K Buffer][Second V Buffer]
    // time = false;
    fp162 __shared__ KVs[2][2][BX * D];
    fp162 regQ[1 * D];
    fp162 regO[1 * D] = {};
    fp162 s2;
    #ifdef TIME
    long long int start;
    #endif
    float l = 0;
    constexpr uint NUM_THREADS = BY;
    //Shuffle Q pointer to the right place
    Q += blockIdx.x * BY * D;
    O += blockIdx.x * BY * D;

    //thread gets its job
    const uint tix = threadIdx.x;

    // //Scuffed clever loading
    // for (uint loadQBXBlock = 0; loadQBXBlock < BY; loadQBXBlock += 2 * BX)
    // {   
    //     //Load a BXxD Chunk of Q into KVs
    //     loadGMEMToSMEM<NUM_THREADS, 4 * BX * D>(Q, &KVs[0][0][0], block);
    //     __syncthreads();
    //     Q += 4 * D * BX;
    //     //Get that chunk into the appropriate register
    //     //We're eating some nasty SMEM conflicts here
    //     int rowIdx = (tix - loadQBXBlock);
    //     if(rowIdx >= 0 && rowIdx < 4 * BX){
    //         int buffer_idx = rowIdx / (2 * BX);
    //         int KV_idx = (rowIdx % (2 * BX)) / BX;
    //         int reg_idx = (rowIdx % BX) * D;
    //         for (uint i = 0; i < D; i++) regQ[i] = KVs[buffer_idx][KV_idx][reg_idx + i];
    //     }
    //     __syncthreads();
    // }

    //thread works with block to load Q
    //Load Q to SMEM in BX Chunks
    //Load those into regQ
    #ifdef TIME
    if (time && tix == 0){
        start = clock64();
    }
    #endif
    for (uint loadQBXBlock = 0; loadQBXBlock < BY; loadQBXBlock += 2 * BX)
    {   
        //Load a BXxD Chunk of Q into KVs
        loadGMEMToSMEM<NUM_THREADS, 2 * BX * D>(Q, &KVs[0][0][0], block);
        __syncthreads();
        Q += 2 * D * BX;
        //Get that chunk into the appropriate register
        //We're eating some nasty SMEM conflicts here
        int rowIdx = (tix - loadQBXBlock);
        if(rowIdx >= 0 && rowIdx < 2 * BX){
            for (uint i = 0; i < D; i++) regQ[i] = KVs[0][rowIdx / BX][(rowIdx % BX) * D + i];
        }
        __syncthreads();
    }
    #ifdef TIME
    if (time && tix == 0){
        times[blockIdx.x * 8 + 0] = clock64() - start;
        // printf("Q time for block %d: %llu\n", blockIdx.x, times[blockIdx.x * 8 + 0]);
    }
    #endif
    
    //Prefetch first KV Buffer
    loadGMEMToSMEM<NUM_THREADS, BX * D>(K, &KVs[0][0][0], block);
    loadGMEMToSMEM<NUM_THREADS, BX * D>(V, &KVs[0][1][0], block);
    K += BX * D;
    V += BX * D;
    int active_buffer = 0;
    // Loop over X
    //INVARIANT: Enter an iteration with one memcpy async cooking (on active buffer)
    for (uint KVBlock = 0; KVBlock < X; KVBlock += BX)
    {
        
        //Wait for memcpy async from previous iteration
        #ifdef TIME
        if (time && tix == 0){
            start = clock64();
        }
        #endif
        cg::wait(block);
        #ifdef TIME
        if (time && tix == 0){
            times[blockIdx.x * 8 + 1] += clock64() - start;
        }
        #endif
        #ifdef TIME
        if (time && tix == 0){
            start = clock64();
        }
        #endif
        //threads load part of K and V, size BX * D
        loadGMEMToSMEM<NUM_THREADS, BX * D>(K, &KVs[1 - active_buffer][0][0], block);
        loadGMEMToSMEM<NUM_THREADS, BX * D>(V, &KVs[1 - active_buffer][1][0], block);
        //Shuffle K and V forwards
        K += BX * D;
        V += BX * D;

        //Looping over BX
        #pragma unroll
        for (uint resIdx = 0; resIdx < BX; resIdx+=1)
        {
            //Initialise accumulator to zero
            s2 = __half2half2(CUDART_ZERO_FP16);

            //Calculate S = QK^T dot product
            #pragma unroll
            for (uint dotIdx = 0; dotIdx < D; dotIdx++)
            {
                s2 =__hadd2(s2, __hmul2(regQ[dotIdx], KVs[active_buffer][0][resIdx * D + dotIdx]));
            }
            //Combine both parts of S (even and odd components of dot on D-axis), duplicate this
            s2 = __half2half2(__hadd(s2.x, s2.y));
            
            //Calculate O = S V
            #pragma unroll
            for (uint dotIdx = 0; dotIdx < D; dotIdx++)
            {
                regO[dotIdx] = __hadd2(regO[dotIdx], __hmul2(s2, KVs[active_buffer][1][resIdx * D + dotIdx]));
            }
            //Calculate l = sum(s^2)
            float s_flt = __half2float(s2.x);
            l += s_flt * s_flt;
        }
        active_buffer = 1 - active_buffer;
        #ifdef TIME
        if (time && tix == 0){
            times[blockIdx.x * 8 + 2] += clock64() - start;
        }
        #endif
    }
    __syncthreads();
    #ifdef TIME
    if (time && tix == 0){
        start = clock64();
    }
    #endif
    float rsqrt_l = rsqrt(l);
    fp162 norm_coeff = __float2half2_rn(rsqrt_l);
    for (uint yIdx = 0; yIdx < D; yIdx++)
    {
        regO[yIdx] = __hmul2(regO[yIdx], norm_coeff);
    }
    __syncthreads();
    #ifdef TIME
    if (time && tix == 0){
        times[blockIdx.x * 8 + 3] = clock64() - start;
    }
    #endif

    #ifdef TIME
    if (time && tix == 0){
        start = clock64();
    }
    #endif
    for (uint storeOBXBlock = 0; storeOBXBlock < BY; storeOBXBlock += 2 * BX)
    {   
        int rowIdx = (tix - storeOBXBlock);
        if(rowIdx >= 0 && rowIdx < 2 * BX){
            for (int i = 0; i < D; i += 4)
            {
                int write_idx = i;
                reinterpret_cast <float4 *>(&KVs[0][rowIdx / BX][(rowIdx % BX) * D + write_idx])[0] = reinterpret_cast <float4 *>(&regO[write_idx])[0];
            }
        }
        __syncthreads();
        //Load a BXxD Chunk of Q into KVs
        storeSMEMtoGMEM<NUM_THREADS, 2 * BX * D>(&KVs[0][0][0], O, block);
        O += 2 * BX * D;
        __syncthreads();
    }
    #ifdef TIME
    if (time && tix == 0){
        times[blockIdx.x * 8 + 4] = clock64() - start;
    }
    #endif
    
}

template<const int D>
void run_flashsign3_cuda(int X, int Y, fp16 *Q, fp16 *K, fp16 *V, fp16 *O){
    constexpr int D_HALVED = D / 2;
    constexpr uint BY = 128;
    constexpr uint BX = 8;
    dim3 gridDim(ceil_div(Y, BY));
    dim3 blockDim(BY);
    fp162 *Q_half = reinterpret_cast<fp162 *>(Q);
    fp162 *K_half = reinterpret_cast<fp162 *>(K);
    fp162 *V_half = reinterpret_cast<fp162 *>(V);
    fp162 *O_half = reinterpret_cast<fp162 *>(O);
    // cudaFuncSetAttribute(kernel<BX, BY, D_HALVED>, cudaFuncAttributePreferredSharedMemoryCarveout, 30);
    kernel<BX, BY, D_HALVED><<<gridDim, blockDim>>>(X, Q_half, K_half, V_half, O_half);
}

template<const int D>
void run_flashsign_3_pytorch(int X, int Y, fp162 *Q, fp162 *K, fp162 *V, fp162 *O){
    constexpr int D_HALVED = D / 2;
    constexpr uint BY = 128;
    constexpr uint BX = 8;
    dim3 gridDim(ceil_div(Y, BY));
    dim3 blockDim(BY);
    kernel<BX, BY, D_HALVED><<<gridDim, blockDim>>>(X, Q, K, V, O);
}
}

torch::Tensor flashsign_3(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
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
    __half2* Qp = reinterpret_cast<__half2 *>(Q.data_ptr<at::Half>());
    __half2* Kp = reinterpret_cast<__half2 *>(K.data_ptr<at::Half>());
    __half2* Vp = reinterpret_cast<__half2 *>(V.data_ptr<at::Half>());
    __half2* Op = reinterpret_cast<__half2 *>(O.data_ptr<at::Half>());
    
    flashsign_kernel3::run_flashsign_3_pytorch<D>(X, Y, Qp, Kp, Vp, Op);

    return O;
}