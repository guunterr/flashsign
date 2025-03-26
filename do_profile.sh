#! /usr/bin/bash
nvidia-smi
mkdir out
rm out/*
nvcc -arch=sm_70 -g -DNDEBUG --generate-line-info -O3 --ptxas-options=-v -o out/temp temp.cu
./out/temp > output_$(date +%m-%d_%H:%M).txt
ncu -f --set full --export profile_$(date +%m-%d_%H:%M) ./out/temp
