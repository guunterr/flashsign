#include <torch/types.h>
#include <ATen/ATen.h>
#include <cuda_fp16.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void
loadSaveKernel(__half2* source, __half2* target, int width){
    int thread_row = blockDim.x * blockIdx.x + threadIdx.x;
    int thread_col = blockDim.y * blockIdx.y + threadIdx.y;
    target[thread_row*width + thread_col] =  __hmul2(source[thread_row*width + thread_col] , source[thread_row*width + thread_col]);
}

torch::Tensor load_save(torch::Tensor X){
    CHECK_INPUT(X);
    //change
    int m = X.size(0);
    int n = X.size(1);

    // Create matrices according to the number of splits.
    auto O   = torch::zeros({m, n}, X.options());
    // Set up the pointers.
    __half2*  Xp = (__half2*)  X.data_ptr<at::Half>();
    __half2*  Op = (__half2*)  O.data_ptr<at::Half>();
    
    // Make sure its divisible by 16!!
    dim3 blocks(m/16, n/32);
    dim3 threads(16, 16);

    loadSaveKernel<<<blocks, threads>>>(
        Xp, Op, n/2);

    return O;
}