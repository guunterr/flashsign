#! /usr/bin/bash
nvidia-smi
mkdir out
nvcc -arch=sm_70 -g --line-info -O3 --ptxas-options=-v -o out/temp temp.cu
ncu --set full --export profile ./out/temp
./out/temp > output.txt