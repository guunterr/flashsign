#! /usr/bin/bash
nvidia-smi
nvcc -V
mkdir out
nvcc -arch=sm_70 -O3 --ptxas-options=-v -o out/temp temp.cu
./out/temp