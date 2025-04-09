
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdio.h>

#include <chrono>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
#define WARP_SIZE 32
#define NWARPS 16
#define SX 96
#define DSX 24
#define DIM 64
#define SDIM 32
#define WQ 16
#define ASYNC true

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }
namespace cg = cooperative_groups;

#define CUDACHECK(err)                         \
    do {                                       \
        cuda_check((err), __FILE__, __LINE__); \
    } while (false)
inline void cuda_check(cudaError_t error_code, const char* file, int line) {
    if (error_code != cudaSuccess) {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(error_code);
    }
}

class Timer {
   private:
    float elapsedTime = 0.0f;
    float deltaTime;
    char* timerName;
    cudaEvent_t startTime, stopTime;

   public:
    Timer() {
        timerName = "";
    }
    Timer(char* name) {
        timerName = name;
    }
    void print() {
        printf("%s: %fms", timerName, elapsedTime);
    }
    void start() {
        cudaEventCreate(&startTime);
        cudaEventCreate(&stopTime);
        cudaEventRecord(startTime);
    }
    float stop() {
        cudaEventRecord(stopTime);
        cudaEventSynchronize(stopTime);
        cudaEventElapsedTime(&deltaTime, startTime, stopTime);
        elapsedTime += deltaTime;
        return deltaTime;
    }
    float getElapsedTime() {
        return elapsedTime;
    }
};

__device__ void save(__half2* __restrict__ to, __half2* __restrict__ from, const int numel, const int nwarps) {
    __half2 temp;
    if (threadIdx.y < nwarps) {
        int stride = nwarps * WARP_SIZE;
        int shift = threadIdx.y * WARP_SIZE + threadIdx.x;
        for (int i = 0; i < numel / stride; i++) {
            temp = from[i * stride + shift];
            to[i * stride + shift] = temp;
        }
        if (shift < numel % stride) {
            int i = (numel / stride);
            temp = from[i * stride + shift];
            to[i * stride + shift] = temp;
        }
    }
}

// __device__ void attention(__half2& mz, __half2* Ov, __half2* __restrict__ Qv, __half2* sK, __half2* sV, __half2* __restrict__ K, __half2* __restrict__ V, uint xbar, cg::thread_block block){

// }

__device__ void loadAsync(__half2* __restrict__ to, __half2* __restrict__ from, const int numel, cg::thread_block& coop_group) {
    cg::memcpy_async(coop_group, to, from, 2 * numel);
}

__device__ void loadCached(__half2* __restrict__ to, __half2* __restrict__ from, const int numel) {
    int stride = NWARPS * WARP_SIZE;
    int shift = threadIdx.y * WARP_SIZE + threadIdx.x;
    for (int i = 0; i < numel / stride; i++) to[i * stride + shift] = from[i * stride + shift];
    if (shift < numel % stride) to[numel - numel % stride + shift] = from[numel - numel % stride + shift];
}

__device__ void load(__half2* __restrict__ to, __half2* __restrict__ from, const int numel, cg::thread_block& coop_group) {
    (ASYNC)
        ? loadAsync(to, from, numel, coop_group)
        : loadCached(to, from, numel);
}

__device__ void sync(cg::thread_block& coop_group) {
    (ASYNC)
        ? cg::wait(coop_group)
        : __syncthreads();
}

__global__ void
__maxnreg__(255)
    flashKernel(__half2* Q, __half2* K, __half2* V, __half2* O, __half2* mzg, int qbar, int xbar) {
    // assert(blockDim.x == WARP_SIZE);
    // Shared Variables. These are the maximum size we can handle.
    __shared__ __half2 sK[SX * DIM];
    __shared__ __half2 sV[SX * DIM];
    cg::thread_block block = cg::this_thread_block();

    // Local vector
    //__half2 QOv[2][DIM] = {};
    __half2 Qv[DIM];
    __half2 Ov[DIM] = {};
    __half2 mz;

    // g_qbar
    // Indexing with iqbar according to blockIdx.x, group axis qbar by gqbar into bqbar.
    const int gqbar = WARP_SIZE * NWARPS;
    const int iqbar = blockIdx.x * gqbar, bqbar = min(gqbar, qbar - iqbar);
    // t_gqbar
    const int tqbar = WARP_SIZE * threadIdx.y + threadIdx.x;
    // s_gqbar
    // Indexing with jbqbar, stream axis bqbar by SX into sbqbar.
    for (int jbqbar = 0; jbqbar < bqbar; jbqbar += SX) {
        const int sbqbar = min(SX, bqbar - jbqbar);
        load(sK, Q + (iqbar + jbqbar) * DIM, sbqbar * DIM, block);
        sync(block);
        if ((0 <= tqbar - jbqbar) && (tqbar - jbqbar < sbqbar)) {
            for (int i = 0; i < DIM; i++) Qv[i] = sK[(tqbar - jbqbar) * DIM + i];
        }
        block.sync();
    }
    // attention(mz, Ov, Qv, sK, sV, K, V, xbar, block);
    // Local values
    __half2 yh2;
    __half y;
    __half m;
    float z;
    __half mult;
    int x;
    // g_xbar
    // Indexing with ixbar according to blockIdx.y, group axis xbar by gxbar into bxbar.
    const int gxbar = (xbar - 1) / gridDim.y + 1, ixbar = blockIdx.y * gxbar, bxbar = min(gxbar, xbar - ixbar);
    // s_gxbar
    // Indexing with jbxbar, stream axis bxbar by SX into sxbar.
    for (uint jbxbar = 0; jbxbar < bxbar; jbxbar += SX) {
        const int sxbar = min(bxbar - jbxbar, SX);
        load(sK, K + (ixbar + jbxbar) * DIM, sxbar * DIM, block);
        load(sV, V + (ixbar + jbxbar) * DIM, sxbar * DIM, block);
        sync(block);
        if (tqbar < bqbar) {
            for (int k = 0; k < sxbar; k++) {
                yh2 = __half2half2(CUDART_ZERO_FP16);
                for (int i = 0; i < DIM; i += 1) yh2 += Qv[i] * sK[k * DIM + i];
                y = (yh2.x + yh2.y) * __float2half_rn(rsqrtf(2 * DIM));
                if (jbxbar + k == 0) {
                    m = y;
                    z = 1;
                    for (int i = 0; i < DIM; i += 1) Ov[i] += sV[k * DIM + i];
                } else if (y < m) {
                    y = __float2half_rn(expf(__half2float(y - m)));
                    z += (float)y;
                    for (int i = 0; i < DIM; i += 1) Ov[i] += __half2half2(y) * sV[k * DIM + i];
                } else {
                    mult = __float2half_rn(expf(__half2float(m - y)));
                    m = y;
                    z = __half2float(mult) * z + 1;
                    for (int i = 0; i < DIM; i += 1) Ov[i] = __half2half2(mult) * Ov[i] + sV[k * DIM + i];
                }
            }
        }
        block.sync();
    }
    // Indexing with jxbar, stream axis bxbar by SX into dxbar.
    mz = __halves2half2(m, __float2half_rn(z));
    // Save O
    if (gridDim.y == 1) {
        // Indexing with jbqbar, stream axis bqbar by SX into sqbar.
        mzg[blockIdx.y * qbar + (iqbar + tqbar)] = mz;
        for (int jbqbar = 0; jbqbar < bqbar; jbqbar += SX) {
            const int sbqbar = min(bqbar - jbqbar, SX);
            if ((0 <= (tqbar - jbqbar)) && ((tqbar - jbqbar) < sbqbar)) {
                // for (int i=0; i<DIM; i++) sK[(tqbar-jbqbar)*DIM+i]=Qv[i];///__half2half2(mz.y);
                for (int i = 0; i < DIM; i += 1) sK[(tqbar - jbqbar) * DIM + i] = Ov[i] / __half2half2(mz.y);
            }
            block.sync();
            save(O + (iqbar + jbqbar) * DIM, sK, sbqbar * DIM, NWARPS);
            sync(block);
        }
    } else {
        mzg[blockIdx.y * qbar + (iqbar + tqbar)] = mz;
        // Indexing with jbqbar, stream axis bqbar by SX into sqbar.
        for (int jbqbar = 0; jbqbar < bqbar; jbqbar += SX) {
            const int sbqbar = min(bqbar - jbqbar, SX);
            if (0 <= tqbar - jbqbar && tqbar - jbqbar < sbqbar) {
                for (int i = 0; i < DIM; i += 1) sK[(tqbar - jbqbar) * DIM + i] = Ov[i];
            }
            block.sync();
            save(O + (blockIdx.y * qbar + (iqbar + jbqbar)) * DIM, sK, sbqbar * DIM, NWARPS);
            sync(block);
        }
    }
}

int main(){
    dim3 blocks(256, 1);
    dim3 warps(WARP_SIZE, NWARPS);
    flashKernel<<<warps,blocks>>>(0,0,0,0,0,1024,1024);
    return 0;
}

// // The LoadFlash code.
// // Loads a tensor A to all SMEMs.
// std::vector<torch::Tensor> flashAttention(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
//     CHECK_INPUT(Q);
//     CHECK_INPUT(K);
//     CHECK_INPUT(V);
//     Timer myTimer = Timer("Load Time");
//     int xbar = K.size(0);
//     int qbar = Q.size(0);
//     int dataSize = 8 * SX * DIM;

//     // Derive the block sizes.
//     int N_blocks;
//     cudaDeviceGetAttribute(&N_blocks, cudaDevAttrMultiProcessorCount, 0);
//     // if (NWARPS < 7) N_blocks *= 2;
//     // How many blocks do we need?
//     uint Nqbar = (qbar - 1) / (NWARPS * WARP_SIZE) + 1;
//     uint Nxbar = N_blocks * (Nqbar / N_blocks + 1) / Nqbar;
//     // uint Nqbar = (qbar-1)/(NWARPS*WARP_SIZE)+1;
//     // uint Nxbar = 1;
//     dim3 blocks(Nqbar, Nxbar);
//     // Create matrices according to the number of splits.
//     auto O = torch::zeros({Nxbar * qbar, 2 * DIM}, Q.options());
//     auto mzg = torch::zeros({Nxbar * qbar, 2}, Q.options());
//     // Set up the pointers.
//     __half2* Qp = (__half2*)Q.data_ptr<at::Half>();
//     __half2* Kp = (__half2*)K.data_ptr<at::Half>();
//     __half2* Vp = (__half2*)V.data_ptr<at::Half>();
//     __half2* Op = (__half2*)O.data_ptr<at::Half>();
//     __half2* mzgp = (__half2*)mzg.data_ptr<at::Half>();

//     // cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
//     // Lets see if we can assign a value...
//     //__half myHalf = __half(0.2781);
//     // printf("mH: %f, Q0: %f, Q1: %f \n", myHalf, Q[0][0], Q[0][1]);
//     // printf("mH: %x, Q0: %x, Q1: %x \n", myHalf, Q[0][0], Q[0][1]);
//     // printf("mH: %x, Q0: %x, Q1: %x \n", myHalf, Q.data_ptr<at::Half>(), Q.data_ptr<at::Half>()+1);
//     // printf("mH: %x, Q0: %x, Q1: %x \n", myHalf, (void*)(Q.data_ptr<at::Half>()), (void*)(Q.data_ptr<at::Half>()+1));
//     //
//     // std::cout << typeid(Q.data_ptr<__half2>()[0]).name() << ',' << typeid(myHalf).name() << '\n';
//     // Op[0] = firstO;
//     // We have 32 threads per warp, and 12 = nwarps per block.
//     dim3 warps(WARP_SIZE, NWARPS);

//     // Return critical information:
//     //  the elapsed time
//     //  the data size
//     //  the number of blocks
//     //  the threads per block
//     void (*ptr)() = (void (*)())(flashKernel);
//     cudaFuncAttributes attrib;
//     cudaError_t err = cudaFuncGetAttributes(&attrib, ptr);
//     printf("result: %s, numRegs: %d\t, sharedSizeBytes: %d\t, maxDynamicSharedSizeBytes: %d\t, maxThreadsPerBlock: %d\n",
//            cudaGetErrorString(err), attrib.numRegs, attrib.sharedSizeBytes, attrib.maxDynamicSharedSizeBytes, attrib.maxThreadsPerBlock);

//     int smemSize, totalRegs;
//     cudaDeviceGetAttribute(&smemSize, cudaDevAttrMaxSharedMemoryPerBlock, 0);
//     cudaDeviceGetAttribute(&totalRegs, cudaDevAttrMaxRegistersPerBlock, 0);
//     printf("smemSize: %d\t cudaDevAttrMaxRegistersPerBlock: %d\n", smemSize, totalRegs);
//     // cudaStream_t stream;
//     // cudaStreamAttrValue stream_attribute;
//     // stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(Op);
//     // stream_attribute.accessPolicyWindow.num_bytes = O.numel()*sizeof(__half);
//     // stream_attribute.accessPolicyWindow.hitRatio = 1.0;
//     // stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
//     // stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
//     // cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);

//     myTimer.start();
//     flashKernel<<<blocks, warps>>>(
//         Qp, Kp, Vp, Op, mzgp, qbar, xbar);
//     // cudaDeviceSynchronize();
//     myTimer.stop();
//     // printf("%d, %d, %d, %d\n", K.numel(), V.numel(), Q.numel(), O.numel());
//     // cudaError_t error = cudaGetLastError();
//     // if (error != cudaSuccess){
//     // fprintf(stderr, "%s\n", cudaGetErrorString(error));
//     // fflush(stderr);
//     // }
//     // CUDACHECK(cudaPeekAtLastError());

//     auto output = torch::ones({3});
//     output[0] = myTimer.getElapsedTime();
//     output[1] = ((2 * K.numel() + 2 * V.numel() + 2 * Q.numel() + 2 * O.numel()));
//     output[2] = (blocks.x * blocks.y);
//     return {output, O, mzg};
// }