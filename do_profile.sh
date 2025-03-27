#! /usr/bin/bash
nvidia-smi
mkdir out
rm out/*
nvcc -arch=sm_70 -g -DNDEBUG --generate-line-info -O3 --ptxas-options=-v -o out/temp temp.cu
./out/temp t $1 1 1 > out/test_$(date +%m-%d_%H:%M).txt
./out/temp r $1 1 3 > out/benchmark_$(date +%m-%d_%H:%M).txt
ncu -f --launch-skip 1 --set full --export out/profile_$(date +%m-%d_%H:%M) ./out/temp r $1 1 1
