#pragma once

std::default_random_engine generator = std::default_random_engine(time(0));


typedef __half fp16;

fp16 f2fp16(float x){
    return __float2half(x);
}

float fp162f(fp16 x){
    return __half2float(x);
}

void randomise_matrix(fp16 *matrix, int N) {
    std::normal_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < N; i++) {
        matrix[i] = f2fp16(distribution(generator));
    }
}

void initialise_matrix(fp16 *matrix, int M, int N) {
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i*N+ j] = f2fp16(((float)(i +  j)) / 10000.0f);
        }
        
    }
    
}

void verify_matrix(fp16 *A, fp16 *B, int M, int N, fp16 epsilon = 0.05){
    fp16 diff = 0;
    for (int i = 0; i < M * N; i++)
    {
        diff = __habs(A[i] - B[i]);
        if (diff > epsilon)
        {
            printf("Matrices disagree at (%d, %d) by %.3f, with values %.3f, %.3f\n", i/N, i%N, __half2float(diff), __half2float(A[i]), __half2float(B[i]));
        }
    }
    printf("Matrices agree to within %.5f\n", __half2float(epsilon));
}

__global__ void print_device_matrix(fp16 *matrix, int M, int N) {
    if(threadIdx.x == 0 && blockIdx.x == 0) printf("%f\n", __half2float(matrix[13]));
}

int ceil_div(int a, int b) {
    return (a / b) + (a % b != 0);
}