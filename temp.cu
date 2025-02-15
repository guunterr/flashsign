#include <iostream>
#include <cmath>
// Kernel function to add the elements of two arrays

__global__ void hello(){
  printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

__global__ void add(float *a, float *b, int n){
  for (int i = 0; i < n; i++)
  {
    a[i] = a[i] + b[i];
  }
  
}

int main(void){

  float *a, *b, *d_a, *d_b;
  int N = 1<<12;
  a = (float *)malloc(N*sizeof(float));
  b = (float *)malloc(N*sizeof(float));
  for(int i = 0; i < N; i++){
    a[i] = i;
    b[i] = 2 * i;
  }
  cudaMalloc((void**)&d_a, N*sizeof(float));
  cudaMemcpy(d_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_b, N*sizeof(float));
  cudaMemcpy(d_b, b, N*sizeof(float), cudaMemcpyHostToDevice);

  add<<<1, 1>>>(d_a, d_b, N);
  cudaDeviceSynchronize();
  cudaMemcpy(a, d_a, N*sizeof(float), cudaMemcpyDeviceToHost);
  float error = 0;
  for(int i = 0; i < N; i++){
    error += fabs(a[i] - 3 * i);
  }
  printf("Error: %f\n", error);
  cudaFree(d_a);
  cudaFree(d_b);
  free(a);
  free(b);
  return 0;
}