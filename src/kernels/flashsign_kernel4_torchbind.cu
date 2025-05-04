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