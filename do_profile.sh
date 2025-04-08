#! /usr/bin/bash
nvidia-smi
mkdir out
rm out/*
datetime=$2
nvcc -arch=sm_70 -lcublas -g -DNDEBUG --generate-line-info -O3 --ptxas-options=-v -o out/temp temp.cu > out/compiler_log_$datetime.txt
# ./out/temp t $1 256 1 > out/test_$datetime.txt
./out/temp r $1 2 20 > out/benchmark_$datetime.txt
ncu -f --launch-skip 1 --set full --export out/profile_$datetime ./out/temp r $1 1 1
