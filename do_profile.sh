#! /usr/bin/bash
nvidia-smi
mkdir out
rm out/*
nvcc -arch=sm_70 -g -DNDEBUG --generate-line-info -O3 --ptxas-options=-v -o out/temp temp.cu
./out/temp > out/output_$(date +%m-%d_%H:%M).txt
ncu -f --set full --export out/profile_$(date +%m-%d_%H:%M) ./out/temp
