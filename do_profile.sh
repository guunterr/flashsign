#! /usr/bin/bash
nvidia-smi
nvcc -V
nvcc -arch=sm_75 -O3 --ptxas-options=-v -o temp temp.cu
./temp