#! /usr/bin/bash
nvidia-smi
mkdir out
nvcc -arch=sm_70 -g --generate-line-info -O3 --ptxas-options=-v -o out/temp temp.cu
./out/temp > output.txt
# ncu -f --set full --export profile ./out/temp
