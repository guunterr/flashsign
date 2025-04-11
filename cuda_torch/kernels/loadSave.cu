#include <torch/types.h>
#include <ATen/ATen.h>
#include "cuda_fp16.h"


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void
loadSaveKernel(__half2* source, __half2* target, int width){
    int thread_x = 16*blockIdx.x + threadIdx.x;
    int thread_y = 16*blockIdx.y + threadIdx.y;
    target[thread_x*width + thread_y] =  source[thread_x*width + thread_y] * source[thread_x*width + thread_y];
}

torch::Tensor load_save(torch::Tensor X){
    CHECK_INPUT(X);
    //change
    int width = X.size(0);
    int depth = X.size(1);

    // Create matrices according to the number of splits.
    auto O   = torch::zeros({width, depth}, X.options());
    // Set up the pointers.
    __half2*  Xp = (__half2*)  X.data_ptr<at::Half>();
    __half2*  Op = (__half2*)  O.data_ptr<at::Half>();
    
    // Make sure its divisible by 16!!
    dim3 blocks(width/16, depth/32);
    dim3 threads(16, 16);

    loadSaveKernel<<<blocks, threads>>>(
        Xp, Op, width);

    return O;
}