#! /usr/bin/bash
nvidia-smi
mkdir out
rm out/*
nvcc -arch=sm_70 -g -DNDEBUG --generate-line-info -O3 --ptxas-options=-v -o out/temp temp.cu
./out/temp 4 1 3 > out/output_$(date +%m-%d_%H:%M).txt
ncu -f --launch-skip 1 --set full --export out/profile_$(date +%m-%d_%H:%M) ./out/temp 4 1 1
