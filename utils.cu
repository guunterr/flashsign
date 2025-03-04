float random_float() {
    float tmp = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) / 5.0f);
    return rand() % 2 == 0 ? tmp : tmp * (-1);
}

void randomise_matrix(float *matrix, int N) {
    for (int i = 0; i < N; i++) {
        matrix[i] = random_float();
    }
}

void copy_matrix(float *src, float *dst, int N) {
    for (int i = 0; i < N; i++) {
        dst[i] = src[i];
    }
}

float *make_matrix_copy(float *matrix, int N) {
    float *copy = (float *)malloc(N * sizeof(float));
    copy_matrix(matrix, copy, N);
    return copy;
}