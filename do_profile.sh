#! /usr/bin/bash
nvidia-smi
nvcc -V
mkdir out
nvcc -arch=sm_70 -O3 --ptxas-options=-v -o out/temp temp.cu
ncu --set full --export profile --kernel "kernel_4" ./out/temp
./out/temp > output.txt