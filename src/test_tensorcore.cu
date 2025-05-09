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
#define ldmatrix1_t(m, src) asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];" : "=r"(m) : "l"(src))

#define ldmatrix2(r0, r1, src) asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(r0), "=r"(r1) : "l"(src))
#define ldmatrix2_t(r0, r1, src) asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];" : "=r"(r0), "=r"(r1) : "l"(src))

#define ldmatrix4(r0, r1, r2, r3, src) asm volatile(                                    \
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16"                                          \
    "{%0, %1, %2, %3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "l"(src)       \
)
// #define ldmatrix4(m, src) asm volatile("ldmatrix.sync.")

#define mma_m16n8k8_fp16(a0, a1,b,c0, c1,d0, d1)                            \
    asm volatile(   "{\n\t"                                                 \
                    ".reg .f16x2 %%a<2>, %%b, %%c<2>, %%d<2>; \n\t"         \
                    "mov.b32 %%a0, %5; \n\t"                                \
                    "mov.b32 %%a1, %6; \n\t"                                \
                    "mov.b32 %%b, %2; \n\t"                                 \
                    "mov.b32 %%c0, %3; \n\t"                                \
                    "mov.b32 %%c1, %4; \n\t"                                \
                    "mov.b32 %%d0, %0; \n\t"                                \
                    "mov.b32 %%d1, %1; \n\t"                                \
                    "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16"      \
                    "   {%%d0, %%d1},"                                      \
                    "   {%%a0, %%a1},"                                      \
                    "   {%%b},"                                             \
                    "   {%%c0, %%c1}; \n\t"                                 \
                    "mov.b32 %0, %%d0; \n\t"                                \
                    "mov.b32 %1, %%d1; \n \t"                               \
                    "}"                                                     \
                    : "=r"(d0), "=r"(d1)                                    \
                    : "r"(b), "r"(c0), "r"(c1), "r"(a0), "r"(a1)            \
                )

#define mma_m16n8k16_fp16(a0, a1, a2, a3, b0, b1, c0, c1, d0, d1)           \
    asm volatile(   "{\n\t"                                                 \
                    ".reg .f16x2 %%a<4>, %%b<2>, %%c<2>, %%d<2>; \n\t"      \
                    "mov.b32 %%d0, %0; \n\t"                                \
                    "mov.b32 %%d1, %1; \n\t"                                \
                    "mov.b32 %%a0, %2; \n\t"                                \
                    "mov.b32 %%a1, %3; \n\t"                                \
                    "mov.b32 %%a2, %4; \n\t"                                \
                    "mov.b32 %%a3, %5; \n\t"                                \
                    "mov.b32 %%b0, %6; \n\t"                                \
                    "mov.b32 %%b1, %7; \n\t"                                \
                    "mov.b32 %%c0, %8; \n\t"                                \
                    "mov.b32 %%c1, %9; \n\t"                                \
                    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"     \
                    "   {%%d0, %%d1},"                                      \
                    "   {%%a0, %%a1, %%a2, %%a3},"                          \
                    "   {%%b0, %%b1},"                                      \
                    "   {%%c0, %%c1}; \n\t"                                 \
                    "mov.b32 %0, %%d0; \n\t"                                \
                    "mov.b32 %1, %%d1; \n \t"                               \
                    "}"                                                     \
                    : "=r"(d0), "=r"(d1)                                    \
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), \
                    "r"(c0), "r"(c1)                                        \
                )

__device__ float2 unpack_half2_cvt_float2(uint32_t packed_floats){
    fp162 floats = reinterpret_cast<fp162*>(&packed_floats)[0];
    return __half22float2(floats);
}

__device__ int2 lane_idx_in_frag(int lane_id){
    int2 out;
    out.x = lane_id / 4;
    out.y = 2 * (lane_id % 4);
    return out;
}

__device__ void write_fragment(uint32_t fragment, fp16* write_ptr){
    uint32_t* write_ptr_32 = reinterpret_cast<uint32_t*>(write_ptr);
    write_ptr_32[0] = fragment;
}

template<const int BX, const int BY, const int D>
__global__ void kernel(int X, fp16* Q, fp16* K, fp16* V, fp16* O, long long int * global_timers) {
    //Algorithm -> Load Q -> Prefetch KV Row -> Load Next KV Row, Process current dot product and accumualate l2 -> Normalise O -> Output
    const int tix = threadIdx.x;
    const int warpIdx = tix / WARP_SIZE;
    const int laneIdx = tix % WARP_SIZE;
    const int bufferPadding = 8;
    const int D_Frag_Count = D/8;
    __shared__ fp16 KVBuffer[2][2][BX][D + bufferPadding];

    //2 elements = 1 register per 8x8 fragment, D_Frag_Count needed to span D=128, 4 rows required to fill a warp
    //64 registers just for this!
    uint32_t Q_frags[4][D_Frag_Count] = {};
    uint32_t K_frags[2] = {};
    uint32_t V_frags[2] = {};
    uint32_t S_frags[4][4] = {};
    uint32_t O_frags[4][D_Frag_Count] = {};
    float z;
    
    fp16* Q_start = Q + blockIdx.x * BY * D;
    fp16* O_start = O + blockIdx.x * BY * D;

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

    //Load Q into registers
    fp16* base_pointer = &KVBuffer[0][0][warpIdx * WARP_SIZE][0];
    for (int D_slice = 0; D_slice < D_Frag_Count; D_slice++)
    {
        fp16* read_ptr = &KVBuffer[0][0][tix][D_slice * 8];
        uint64_t read_addr;
        cvta_shared_64(read_addr, read_ptr);
        ldmatrix4(Q_frags[0][D_slice], Q_frags[1][D_slice], Q_frags[2][D_slice], Q_frags[3][D_slice], read_addr);
    }

    __syncthreads();

    int active_buffer = 0;

    for (int KV_Block = 0; KV_Block < X; KV_Block += BX)
    {
        //load KV Block
        fp16* K_base_ptr = K + KV_Block * D;
        for (int K_row = 0; K_row < 32; K_row += access_size)
        {
            fp16* read_ptr = K_base_ptr + K_row * D + tix*access_size;
            fp16* write_ptr = &KVBuffer[active_buffer][0][K_row + (tix/threads_per_row)][(tix % threads_per_row) * access_size];
            uint64_t write_addr;
            cvta_shared_64(write_addr, write_ptr);
            load8_async_prefetch(read_ptr,write_addr);
        }
        fp16* V_base_ptr = V + KV_Block * D;
        for (int V_row = 0; V_row < 32; V_row += access_size)
        {
            fp16* read_ptr = V_base_ptr + V_row * D + tix*access_size;
            fp16* write_ptr = &KVBuffer[active_buffer][1][V_row + (tix/threads_per_row)][(tix % threads_per_row) * access_size];
            uint64_t write_addr;
            cvta_shared_64(write_addr, write_ptr);
            load8_async_prefetch(read_ptr,write_addr);
        }
        commit_group();
        //wait for KV block to load
        wait_all();
        __syncthreads();
        //Zero out accumulator
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                S_frags[i][j] = 0;
            }
        }
        //Perform S = QK^T
        for (int k_chunk = 0; k_chunk < 4; k_chunk++)
        {
            for (int d_chunk = 0; d_chunk < D_Frag_Count; d_chunk+=2)
            {
                fp16* read_ptr = &KVBuffer[active_buffer][0][k_chunk * 8 + laneIdx % 8][8 * (d_chunk + laneIdx / 8)];
                uint64_t read_addr;
                cvta_shared_64(read_addr, read_ptr);
                ldmatrix2(K_frags[0], K_frags[1], read_addr);
                mma_m16n8k16_fp16(Q_frags[0][d_chunk], Q_frags[1][d_chunk], Q_frags[0][d_chunk + 1], Q_frags[1][d_chunk + 1], K_frags[0], K_frags[1], S_frags[0][k_chunk], S_frags[1][k_chunk], S_frags[0][k_chunk], S_frags[1][k_chunk]);
                mma_m16n8k16_fp16(Q_frags[2][d_chunk], Q_frags[3][d_chunk], Q_frags[2][d_chunk + 1], Q_frags[3][d_chunk + 1], K_frags[0], K_frags[1], S_frags[2][k_chunk], S_frags[3][k_chunk], S_frags[2][k_chunk], S_frags[3][k_chunk]);
            }
        }
        //Perform O += SV
        for (int d_chunk = 0; d_chunk < D_Frag_Count; d_chunk++)
        {
            for (int x_chunk = 0; x_chunk < 2; x_chunk++)
            {
                fp16* read_ptr = &KVBuffer[active_buffer][1][x_chunk * 16 + laneIdx][8 * d_chunk];
                uint64_t read_addr;
                cvta_shared_64(read_addr, read_ptr);
                ldmatrix2_t(V_frags[0], V_frags[1], read_addr);
                mma_m16n8k16_fp16(S_frags[0][2 * x_chunk], S_frags[1][2 * x_chunk], S_frags[0][2 * x_chunk + 1], S_frags[1][2 * x_chunk + 1], V_frags[0], V_frags[1], O_frags[0][d_chunk], O_frags[1][d_chunk], O_frags[0][d_chunk], O_frags[1][d_chunk]);
                mma_m16n8k16_fp16(S_frags[2][2 * x_chunk], S_frags[3][2 * x_chunk], S_frags[2][2 * x_chunk + 1], S_frags[3][2 * x_chunk + 1], V_frags[0], V_frags[1], O_frags[2][d_chunk], O_frags[3][d_chunk], O_frags[2][d_chunk], O_frags[3][d_chunk]);
            }
            
        }
        __syncthreads();
        for (int S_buffer_i = 0; S_buffer_i < 4; S_buffer_i++)
        {
            for (int S_buffer_j = 0; S_buffer_j < 4; S_buffer_j++)
            {
                fp16* S_rearrage_write_ptr = &KVBuffer[active_buffer][0][S_buffer_i * 8 + (laneIdx / 4)][warpIdx * 32 + S_buffer_j * 8 + 2 * (laneIdx % 4)];
                write_fragment(S_frags[S_buffer_i][S_buffer_j], S_rearrage_write_ptr);
            }
        }
        __syncthreads();
        for (int i = 0; i < BX; i++)
        {
            fp16 s = KVBuffer[active_buffer][0][laneIdx][warpIdx * 32 + i];
            z += __half2float(__hmul(s,s));
        }
    }

    z = rsqrt(z);
    fp16 l = __float2half(z);
    //Write out O
    //Buffer write though smem to coalesce global writes
    for (int ORowBlock = 0; ORowBlock < 4; ORowBlock++)
    {
        for (int OColBlock = 0; OColBlock < D_Frag_Count; OColBlock++)
        {
            fp16* write_ptr = &KVBuffer[warpIdx / 2][warpIdx % 2][ORowBlock * 8 + (laneIdx / 4)][OColBlock * 8 + 2 * (laneIdx % 4)];
            write_fragment(O_frags[ORowBlock][OColBlock], write_ptr);
        }
    }
    __syncthreads();
    for (int DIdx = 0; DIdx < D; DIdx++)
    {
        KVBuffer[warpIdx/2][warpIdx%2][laneIdx][DIdx] = __hmul(KVBuffer[warpIdx/2][warpIdx%2][laneIdx][DIdx], l);
    }
    __syncthreads();
    //8 elements per thread = 8 rows across warpgroup
    for (int O_row = 0; O_row < BY; O_row+=8)
    {
        fp16* write_ptr = O_start + O_row * D + tix * 8;   
        fp16* read_ptr = &KVBuffer[0][0][O_row + (tix/(D/8))][(tix % (D/8)) * 8];
        
        reinterpret_cast<float4 *>(write_ptr)[0] = reinterpret_cast<float4 *>(read_ptr)[0];
    }
    
}




void run_kernel(int X, int Y, fp16* Q, fp16* K, fp16* V, fp16* O) {
    constexpr uint BY = 128;
    constexpr uint BX = 32;
    constexpr uint D = 128;

    dim3 gridDim(ceil_div(Y, BY));
    dim3 blockDim(BY);
    kernel<BX, BY, D><<<gridDim, blockDim>>>(X, Q, K, V, O, 0);
}
}

int main(int argc, char* argv[]) {
    //Allocate and initialise Q, K, V, O
    constexpr int D = 128;
    const int X = 82944;
    const int Y = 82944;
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
    test_tensorcore::run_kernel(X, Y, d_Q, d_K, d_V, d_O);
    cudaDeviceSynchronize();
    cudaMemcpy(O, d_O, Y * D * sizeof(fp16), cudaMemcpyDeviceToHost);
    
    return 0;

}
