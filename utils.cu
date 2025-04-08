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

int ceil_div(int a, int b) {
    return (a / b) + (a % b != 0);
}