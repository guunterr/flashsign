## GOALS

- Figure out the neural string diagrams
    - Work through the paper and repeat calculations and diagrams
    - Understand what the fuck he means by indexing
    - Reprove the theorems
    - Figure out how to replicate the performance model calculations
    - Try to replicate SGAttention optimisations without just copying his homework
    - Maybe implement some automatic constraint optimisation stuff for architectural search
- Learn how to write CUDA and get stuff working
    - Implement decent saxpy or GEMM
        - Speed this up to somewhat near cublas rate
    - Implement an autotuning hyperparameter search for kernels
- Get a GPU on Vast.ai
    - Find an H100 I can rent
    - Set up an instance to do my programming
    - Set up infrastructure for running tests and benchmarks
    - Set up infrastructure for hyperparameter search on algorithms
- FlashAttention
    - Set up a git repo
    - Iteratively work up to implementing what's in the papers vincent has sent me
- SGAttention
    - Write out vincent's naive diagram into cuda
    - Write out better diagrams into cuda
    - Benchmark all of these for the paper
    - Iteratively improve algorithm until we are >10x faster than pytorch

- SGEMM Kernels
    - Remember to use -O3 LOL
    - BF16 only makes sense when using wmma
    - Kernel 1 60 GFLOPS (BAD!)
    - Kernel 2 570 GFLOPS
    - Kernel 3 960 GFLOPS
    - Kernel 4 1200 GFLOPS
    - I think I need a profiler to keep going...

- Flashsign stuff
    - Compile and run on A100
    - Get 256 thread size working
    - Get FP16 utilisation to 100%